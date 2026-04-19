import logging
import os
from contextlib import asynccontextmanager
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


class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str
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
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.5.0", lifespan=lifespan)

# ── Параметры retrieval ──────────────────────────────────────────────────────
#
# Стратегия: Multi-Sparse Prefetch + Two-Stage Rerank
#
# Dense (1 API-вызов, тарифицируется):
#   └─ 1 prefetch ветка, ловит семантическую близость
#
# Sparse BM25 (локально, бесплатно, N веток):
#   ├─ main_enriched  — основной запрос + keywords + entities
#   ├─ keywords_only  — чистый лексический матч по ключевым словам
#   ├─ variant_0      — альтернативная формулировка вопроса
#   ├─ variant_1      — ещё одна формулировка
#   └─ hyde_0         — первая строка гипотетического документа
#
# Все ветки сливаются Qdrant-RRF → широкий пул кандидатов.
# Реранкер финально сортирует топ-N по семантической релевантности.
#
DENSE_PREFETCH_K = 20       # кандидатов от dense-ветки

# Лимиты для каждой sparse-ветки (разные, т.к. разная точность источника)
SPARSE_MAIN_K = 50          # основной обогащённый запрос — самый надёжный
SPARSE_KEYWORDS_K = 40      # только ключевые слова — высокая точность терминов
SPARSE_VARIANT_K = 30       # варианты вопроса — расширяют лексику
SPARSE_HYDE_K = 25          # первая строка HyDE — гипотетический ответ

RETRIEVE_K = 60             # итоговый пул после RRF (больше = выше Recall@50)
RERANK_LIMIT = 20           # реранкер видит топ-20 → хороший nDCG

DENSE_QUERY_MAX_CHARS = 512


def build_dense_query(question: Question) -> str:
    """
    Один компактный текст для dense embedding.
    Ровно 1 HTTP-запрос к API на вопрос — rate limit соблюдён.
    """
    base = question.search_text.strip() if question.search_text.strip() else question.text.strip()

    entity_tokens: list[str] = []
    if question.entities:
        if question.entities.people:
            entity_tokens.extend(question.entities.people)
        if question.entities.names:
            entity_tokens.extend(question.entities.names)

    combined = f"{base}\n{' '.join(entity_tokens)}" if entity_tokens else base

    if len(combined) > DENSE_QUERY_MAX_CHARS:
        combined = combined[:DENSE_QUERY_MAX_CHARS]

    logger.info("Dense query (%d chars): %r", len(combined), combined[:120])
    return combined


def build_sparse_queries(question: Question) -> list[tuple[str, str, int]]:
    """
    Возвращает список (name, text, prefetch_limit) для каждой sparse-ветки.
    Все вычисляются локально — rate limit не затрагивается.

    Принцип каждой ветки:
      main_enriched — широкий охват: base + keywords + entities + date_mentions
      keywords_only — точный терминологический матч без «шума» вопроса
      variant_N     — альтернативные формулировки от препроцессора вопросов
      hyde_0        — первая строка гипотетического документа-ответа
    """
    base = question.search_text.strip() if question.search_text.strip() else question.text.strip()

    # Собираем переиспользуемые части
    entity_tokens: list[str] = []
    if question.entities:
        for field in (question.entities.people, question.entities.names, question.entities.documents):
            if field:
                entity_tokens.extend(field)

    keyword_str = " ".join(question.keywords) if question.keywords else ""
    date_str = " ".join(question.date_mentions) if question.date_mentions else ""

    branches: list[tuple[str, str, int]] = []

    # main enriched
    main_parts = [base]
    if keyword_str:
        main_parts.append(keyword_str)
    if entity_tokens:
        main_parts.append(" ".join(entity_tokens))
    if date_str:
        main_parts.append(date_str)
    branches.append(("main_enriched", "\n".join(filter(None, main_parts)), SPARSE_MAIN_K))

    # keywords only
    # Если keywords нет — строим из entities, чтобы ветка не пустовала
    keywords_text = " ".join(filter(None, [keyword_str, " ".join(entity_tokens)]))
    if keywords_text.strip():
        branches.append(("keywords_only", keywords_text, SPARSE_KEYWORDS_K))

    # варианты вопроса
    if question.variants:
        for i, variant in enumerate(question.variants[:2]):
            v = variant.strip()
            if v:
                # Обогащаем вариант keywords для лучшего покрытия
                v_text = f"{v}\n{keyword_str}" if keyword_str else v
                branches.append((f"variant_{i}", v_text, SPARSE_VARIANT_K))

    # HyDE первая строка
    if question.hyde:
        first_line = question.hyde[0].strip().split("\n")[0][:300]
        if first_line:
            branches.append(("hyde_0", first_line, SPARSE_HYDE_K))

    logger.info(
        "Sparse branches: %s",
        [(name, len(text), k) for name, text, k in branches],
    )
    return branches


async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
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


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    "Батчевый sparse embedding — один вызов модели для всех текстов."
    model = get_sparse_model()
    result: list[SparseVector] = []
    for item in model.embed(texts):
        result.append(
            SparseVector(
                indices=[int(i) for i in item.indices.tolist()],
                values=[float(v) for v in item.values.tolist()],
            )
        )
    return result


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_branches: list[tuple[str, SparseVector, int]],  # (name, vector, limit)
) -> list[Any] | None:
    """
    Multi-branch hybrid search.

    Схема prefetch:
      [dense_main] + [sparse_main] + [sparse_keywords] + [sparse_variant_0]
                  + [sparse_variant_1] + [sparse_hyde_0]
                          │
                    RRF fusion (Qdrant)
                          │
                    top-RETRIEVE_K кандидатов

    Больше веток = больше разнообразия кандидатов = выше Recall@50.
    Каждая ветка независимо ловит чанки, которые другие могут пропустить.
    """
    prefetches: list[models.Prefetch] = []

    # Dense ветка (1 штука — rate limit)
    prefetches.append(
        models.Prefetch(
            query=dense_vector,
            using=QDRANT_DENSE_VECTOR_NAME,
            limit=DENSE_PREFETCH_K,
        )
    )

    # Sparse ветки (N штук — все локальные)
    for name, sparse_vec, limit in sparse_branches:
        prefetches.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=limit,
            )
        )

    logger.info(
        "Qdrant query: %d prefetch branches, retrieve_k=%d",
        len(prefetches),
        RETRIEVE_K,
    )

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )

    if not response.points:
        return None

    return response.points


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
    data = response.json().get("data") or []
    return [float(sample["score"]) for sample in data]


async def rerank_points(
    client: httpx.AsyncClient,
    question: Question,
    points: list[Any],
) -> list[Any]:
    """
    Двухэтапное реранжирование:

    Этап 1 — Reranker API (топ-RERANK_LIMIT кандидатов):
      Семантическая оценка пары (вопрос, текст чанка).
      Используем search_text если есть — он точнее сформулирован.
      Результат: лучшие чанки всплывают наверх → хороший nDCG.

    Этап 2 — Хвост (остальные кандидаты):
      Добавляем в конец в порядке RRF-скора из Qdrant.
      Они не потеряны — влияют на Recall@50 через свои message_ids.

    Итоговый порядок: [reranked_top] + [rrf_tail]
    """
    rerank_candidates = points[:RERANK_LIMIT]
    tail = points[RERANK_LIMIT:]

    # Для реранка берём search_text если есть — он содержательнее
    rerank_query = (
        question.search_text.strip()
        if question.search_text.strip()
        else question.text.strip()
    )

    rerank_targets = [
        (point.payload or {}).get("page_content") or "" for point in rerank_candidates
    ]
    scores = await get_rerank_scores(client, rerank_query, rerank_targets)

    reranked = [
        point
        for _, point in sorted(
            zip(scores, rerank_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    logger.info(
        "Rerank: top=%d score=%.4f, bottom=%d score=%.4f",
        1, max(scores) if scores else 0,
        len(scores), min(scores) if scores else 0,
    )

    return reranked + tail


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

    dense_query = build_dense_query(question)
    sparse_query_branches = build_sparse_queries(question)

    # API-вызов (rate limit)
    dense_vector = await embed_dense(http, dense_query)

    # Sparse: батчем через локальную модель (все тексты за один проход)
    sparse_texts = [text for _, text, _ in sparse_query_branches]
    sparse_vectors = embed_sparse_texts(sparse_texts)

    sparse_branches = [
        (name, vec, limit)
        for (name, _, limit), vec in zip(sparse_query_branches, sparse_vectors)
    ]

    # Multi-branch retrieval
    best_points = await qdrant_search(qdrant, dense_vector, sparse_branches)

    if best_points is None:
        logger.info("No points found for: %r", original_text)
        return SearchAPIResponse(results=[])

    logger.info("Retrieved %d candidate points", len(best_points))

    # Two-stage reranking
    best_points = await rerank_points(http, question, list(best_points))

    # Сборка message_ids с дедупликацией
    # Порядок важен для nDCG: реранкнутые чанки идут первыми.
    # Каждый чанк содержит несколько message_ids — все попадают в ответ.
    seen: set[str] = set()
    message_ids: list[str] = []
    for point in best_points:
        for mid in extract_message_ids(point):
            if mid not in seen:
                seen.add(mid)
                message_ids.append(mid)

    logger.info("Returning %d unique message_ids", len(message_ids))

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

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()