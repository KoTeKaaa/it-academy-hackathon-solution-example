import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Optional
import asyncio


import httpx
from datetime import datetime
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models, Filter

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def validate_required_env() -> None:
    """Проверяет наличие обязательных переменных окружения"""
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if not os.getenv(name)
    ]
    if missing_env_vars:
        logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
        raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    """Возвращает kwargs для авторизации внешних запросов"""
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


DENSE_PREFETCH_K = 10
SPARSE_PREFETCH_K = 30
RETRIEVE_K = 20
RERANK_LIMIT = 10
FINAL_LIMIT = 50


def build_search_queries(question: Question) -> tuple[str, str]:
    """
    Формирует тексты для dense и sparse эмбеддингов.
    dense: семантический поиск (search_text или text)
    sparse: keyword-поиск (основной текст + keywords)
    """
    dense_query = (question.search_text or question.text).strip()

    sparse_parts = [dense_query]
    if question.keywords:
        sparse_parts.extend(question.keywords[:10])

    return dense_query, " ".join(sparse_parts)

def _iso_to_timestamp(iso_str: str) -> float:
    """Конвертирует ISO-строку в Unix timestamp для совместимости с Qdrant"""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp()


def build_qdrant_filter(question: Question) -> Optional[models.Filter]:
    conditions = []

    if question.date_range:
        conditions.extend([
            models.FieldCondition(
                key="date_start",  # ← без metadata.
                range=models.Range(gte=_iso_to_timestamp(question.date_range.from_)),
            ),
            models.FieldCondition(
                key="date_end",  # ← без metadata.
                range=models.Range(lte=_iso_to_timestamp(question.date_range.to)),
            ),
        ])

    if question.entities and question.entities.people:
        conditions.append(
            models.FieldCondition(
                key="participants",  # ← без metadata.
                match=models.MatchAny(any=question.entities.people),
            )
        )

    if question.entities and question.entities.emails:
        conditions.append(
            models.FieldCondition(
                key="mentions",  # ← без metadata.
                match=models.MatchAny(any=question.entities.emails),
            )
        )

    return models.Filter(must=conditions) if conditions else None


def deduplicate_message_ids(message_ids: list[str], limit: int = FINAL_LIMIT) -> list[str]:
    """Убирает дубликаты message_ids, сохраняя порядок первого вхождения"""
    seen = set()
    result = []
    for mid in message_ids:
        if mid not in seen and len(result) < limit:
            seen.add(mid)
            result.append(mid)
    return result


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    """Загружает sparse-модель с кэшированием (один раз на процесс)"""
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения: создание/закрытие клиентов"""
    app.state.http = httpx.AsyncClient(timeout=60.0)
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
        timeout=60.0,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    """Получает dense-вектор через внешний API"""
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": [text],
            "encoding_format": "float",
        },
    )
    response.raise_for_status()

    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload:
        raise ValueError("Dense embedding response is empty")

    return payload.data[0].embedding


async def embed_sparse(text: str) -> SparseVector:
    """Вычисляет sparse-вектор локально через fastembed"""
    model = get_sparse_model()
    vectors = list(model.embed([text]))

    if not vectors:
        raise ValueError("Sparse embedding response is empty")

    item = vectors[0]
    return SparseVector(
        indices=[int(idx) for idx in item.indices.tolist()],
        values=[float(val) for val in item.values.tolist()],
    )


async def embed_with_expansion(client: httpx.AsyncClient, question: Question) -> list[float]:
    """Получает усреднённый dense-вектор с устойчивостью к частичным ошибкам API"""
    texts = [(question.search_text or question.text).strip()]

    if question.hyde:
        texts.extend(question.hyde[:2])
    if question.variants:
        texts.extend(question.variants[:2])

    # Параллельные запросы с обработкой ошибок
    tasks = [embed_dense(client, t) for t in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_embeddings = [r for r in results if isinstance(r, list)]
    if not valid_embeddings:
        raise RuntimeError("All embedding requests failed")

    dim = len(valid_embeddings[0])
    avg_vector = [
        sum(v[i] for v in valid_embeddings) / len(valid_embeddings)
        for i in range(dim)
    ]
    return avg_vector

async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    query_filter: Optional[models.Filter] = None,
) -> list[Any]:
    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPARSE_PREFETCH_K,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        filter=query_filter,
        with_payload=True,
    )
    return response.points if response.points else []


async def get_rerank_scores(client: httpx.AsyncClient, query: str, targets: list[str]) -> list[float]:
    """Получает скоры релевантности от внешнего rerank-API"""
    if not targets:
        return []

    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": query,
            "text_2": targets,
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []
    return [float(sample["score"]) for sample in data]


async def rerank_points(client: httpx.AsyncClient, query: str, points: list[Any]) -> list[Any]:
    """Переранжирует точки через внешний rerank-API"""
    if not points:
        return []

    rerank_candidates = points[:RERANK_LIMIT]
    rerank_targets = [point.payload.get("page_content", "") for point in rerank_candidates]
    scores = await get_rerank_scores(client, query, rerank_targets)

    return [
        point for _, point in sorted(
            zip(scores, rerank_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
    ]


def extract_message_ids(point: Any) -> list[str]:
    """Извлекает message_ids из payload точки Qdrant"""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return [str(mid) for mid in metadata.get("message_ids", [])]


@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint"""
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    """Основной поисковый эндпоинт"""
    question = payload.question
    query_text = (question.search_text or question.text).strip()

    if not query_text:
        raise HTTPException(status_code=400, detail="question.text is required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    dense_query, sparse_query = build_search_queries(question)

    if question.hyde or question.variants:
        dense_vector = await embed_with_expansion(client, question)
    else:
        dense_vector = await embed_dense(client, dense_query)

    sparse_vector = await embed_sparse(sparse_query)

    query_filter = build_qdrant_filter(question)

    best_points = await qdrant_search(qdrant, dense_vector, sparse_vector, query_filter)
    if best_points is None:
        return SearchAPIResponse(results=[])

    best_points = await rerank_points(client, query_text, list(best_points))

    all_message_ids = []
    for point in best_points:
        all_message_ids.extend(extract_message_ids(point))

    unique_ids = deduplicate_message_ids(all_message_ids, limit=FINAL_LIMIT)

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=unique_ids)]
    )


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Глобальный обработчик исключений"""
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()