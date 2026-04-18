import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

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
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
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

# Метадата чанков в Qdrant'e, по которой вы можете фильтровать
class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str # channel, group, private, thread
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.2.0", lifespan=lifespan)

# Параметры retrieval
DENSE_PREFETCH_K = 15       # топ-K для каждого dense-запроса (в т.ч. hyde)
SPARSE_PREFETCH_K = 40      # топ-K для sparse-запроса
RETRIEVE_K = 30             # итого после RRF-слияния
RERANK_LIMIT = 10           # сколько кандидатов отдаём на реранк
MAX_HYDE_QUERIES = 3        # максимум HyDE-текстов для доп. dense-поиска
MAX_VARIANT_QUERIES = 2     # максимум вариантов вопроса для доп. dense-поиска


# Вспомогательные функции для построения запроса

def build_enriched_query(question: Question) -> str:
    """
    Строим обогащённый текст запроса, объединяя:
    - search_text (если есть) или исходный text
    - keywords (ключевые слова)
    - имена людей и прочие named entities
    """
    # Базовый текст: search_text предпочтительнее (он может быть переформулирован)
    base = question.search_text.strip() if question.search_text.strip() else question.text.strip()

    extras: list[str] = []

    # Ключевые слова улучшают sparse-компоненту
    if question.keywords:
        extras.append(" ".join(question.keywords))

    # Упоминания людей/имён помогают и dense, и sparse
    if question.entities:
        entity_tokens: list[str] = []
        if question.entities.people:
            entity_tokens.extend(question.entities.people)
        if question.entities.names:
            entity_tokens.extend(question.entities.names)
        if entity_tokens:
            extras.append(" ".join(entity_tokens))

    if extras:
        return base + "\n" + " ".join(extras)
    return base


def build_qdrant_filter(question: Question) -> models.Filter | None:
    """
    Формируем must-фильтр для Qdrant на основе метаданных вопроса:
    - date_range  → фильтр по metadata.start / metadata.end
    - entities    → фильтр по metadata.mentions (emails, links)
    """
    conditions: list[Any] = []

    # ── Фильтр по диапазону дат ──────────────────────────────────────────────
    if question.date_range:
        date_filter = _build_date_filter(question.date_range)
        if date_filter:
            conditions.append(date_filter)

    # ── Фильтр по упоминаниям (emails) ───────────────────────────────────────
    # emails в вопросе с высокой вероятностью фигурируют в mentions чанков
    if question.entities and question.entities.emails:
        for email in question.entities.emails:
            conditions.append(
                models.FieldCondition(
                    key="metadata.mentions",
                    match=models.MatchValue(value=email),
                )
            )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def _parse_date_to_timestamp(date_str: str) -> int | None:
    # Пробуем распарсить строку даты в unix-timestamp (UTC).
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def _build_date_filter(date_range: DateRange) -> models.Filter | None:
    """
    Чанк попадает в результат, если его временной диапазон пересекается
    с запрашиваемым: chunk.start <= range.to AND chunk.end >= range.from
    """
    from_ts = _parse_date_to_timestamp(date_range.from_)
    to_ts = _parse_date_to_timestamp(date_range.to)

    if from_ts is None and to_ts is None:
        logger.warning("Could not parse date_range: from=%s to=%s", date_range.from_, date_range.to)
        return None

    sub_conditions: list[Any] = []

    if to_ts is not None:
        # конец чанка должен быть до (или равен) верхней границе
        sub_conditions.append(
            models.FieldCondition(
                key="metadata.date_start",
                range=models.Range(lte=to_ts),
            )
        )

    if from_ts is not None:
        # начало чанка должно быть после (или равно) нижней границе
        sub_conditions.append(
            models.FieldCondition(
                key="metadata.date_end",
                range=models.Range(gte=from_ts),
            )
        )

    if not sub_conditions:
        return None

    return models.Filter(must=sub_conditions)


# Embedding

async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    # Dense endpoint ожидает OpenAI-compatible body с input как списком строк.
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": [text],
        },
    )
    response.raise_for_status()

    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")

    return payload.data[0].embedding


async def embed_dense_batch(
    client: httpx.AsyncClient,
    texts: list[str],
) -> list[list[float]]:
    # Батчевый dense embedding — один HTTP-запрос на все тексты.
    if not texts:
        return []

    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": texts,
        },
    )
    response.raise_for_status()

    payload = DenseEmbeddingResponse.model_validate(response.json())
    # API возвращает элементы в произвольном порядке с полем index — сортируем
    sorted_data = sorted(payload.data, key=lambda d: d.index)
    return [item.embedding for item in sorted_data]


async def embed_sparse(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text]))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")

    item = vectors[0]
    return SparseVector(
        indices=[int(i) for i in item.indices.tolist()],
        values=[float(v) for v in item.values.tolist()],
    )


# Qdrant search

async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vectors: list[list[float]],   # первый — основной запрос, остальные — HyDE/варианты
    sparse_vector: SparseVector,
    query_filter: models.Filter | None = None,
) -> list[Any] | None:
    """
    Hybrid search с поддержкой нескольких dense-векторов (HyDE + варианты).
    Каждый dense-вектор становится отдельным Prefetch, все они сливаются через RRF.
    """
    prefetches: list[models.Prefetch] = []

    # Dense prefetch для каждого вектора (основной + HyDE + варианты)
    for vec in dense_vectors:
        prefetches.append(
            models.Prefetch(
                query=vec,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
                filter=query_filter,
            )
        )

    # Sparse prefetch (один — для основного/обогащённого запроса)
    prefetches.append(
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_PREFETCH_K,
            filter=query_filter,
        )
    )

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
        query_filter=query_filter,
    )

    if not response.points:
        return None

    return response.points


# Reranking

def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    message_ids = metadata.get("message_ids") or []
    return [str(mid) for mid in message_ids]


async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    # Rerank endpoint возвращает score для пары query -> candidate text.
    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": targets,
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []

    return [float(sample["score"]) for sample in data]


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    # Реранжируем топ-RERANK_LIMIT кандидатов, остальные добавляем в конец.
    rerank_candidates = points[:RERANK_LIMIT]
    tail = points[RERANK_LIMIT:]

    rerank_targets = [point.payload.get("page_content") for point in rerank_candidates]
    scores = await get_rerank_scores(client, query, rerank_targets)

    reranked_candidates = [
        point
        for _, point in sorted(
            zip(scores, rerank_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    return reranked_candidates + tail


# Endpoints

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    question = payload.question
    original_text = question.text.strip()

    if not original_text:
        raise HTTPException(status_code=400, detail="question.text is required")

    http: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    # Строим обогащённый запрос
    enriched_query = build_enriched_query(question)
    logger.debug("Enriched query: %r", enriched_query)

    # Собираем тексты для батчевого dense embedding
    # Порядок: [enriched_query, *hyde[:MAX], *variants[:MAX]]
    embed_texts: list[str] = [enriched_query]

    if question.hyde:
        hyde_texts = question.hyde[:MAX_HYDE_QUERIES]
        embed_texts.extend(hyde_texts)
        logger.debug("Adding %d HyDE texts", len(hyde_texts))

    if question.variants:
        variant_texts = question.variants[:MAX_VARIANT_QUERIES]
        embed_texts.extend(variant_texts)
        logger.debug("Adding %d variant texts", len(variant_texts))

    # Dense embedding (один батч-запрос)
    dense_vectors = await embed_dense_batch(http, embed_texts)

    # Sparse embedding (по обогащённому запросу
    sparse_vector = await embed_sparse(enriched_query)

    #  Фильтр по метадате
    query_filter = build_qdrant_filter(question)
    if query_filter:
        logger.debug("Applying Qdrant filter: %s", query_filter)

    # Retrieval
    best_points = await qdrant_search(qdrant, dense_vectors, sparse_vector, query_filter)

    if best_points is None:
        logger.info("No points found for query: %r", original_text)
        return SearchAPIResponse(results=[])

    # Rerank (используем оригинальный вопрос для точности реранка)
    best_points = await rerank_points(http, original_text, list(best_points))

    # Собираем message_ids
    seen: set[str] = set()
    message_ids: list[str] = []
    for point in best_points:
        for mid in extract_message_ids(point):
            if mid not in seen:
                seen.add(mid)
                message_ids.append(mid)

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=message_ids)]
    )


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()