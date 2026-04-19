import logging
import os
from functools import lru_cache
from typing import Any, Optional
import asyncio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str  # group, channel, private
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData


class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    has_forward: bool = False
    has_quote: bool = False
    date_start: Optional[int] = None
    date_end: Optional[int] = None


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


app = FastAPI(title="Index Service", version="0.2.0")

CHUNK_SIZE = 512
OVERLAP_SIZE = 256
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
UVICORN_WORKERS = 8


def render_message(message: Message) -> str:
    """Полный рендер для page_content — сохраняем структуру."""
    if message.is_hidden or message.is_system:
        return ""

    parts = [f"[{message.sender_id}]"]

    if message.is_forward:
        parts.append("[FORWARD]")
    if message.is_quote:
        parts.append("[QUOTE]")
    if message.file_snippets:
        parts.append(f"[FILE]\n{message.file_snippets}")
    if message.text:
        parts.append(message.text)

    if message.parts:
        for part in message.parts:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                media_type = part.get("mediaType")
                if media_type == "forward":
                    parts.append(f"[FORWARD] {part_text}")
                elif media_type == "quote":
                    parts.append(f"[QUOTE] {part_text}")
                else:
                    parts.append(part_text)

    if message.mentions:
        parts.append(f"[MENTIONS] {', '.join(message.mentions)}")

    return "\n".join(filter(None, parts))


def _prepare_base_content(
    message: Message,
    *,
    forward_quote_limit: int,
    regular_part_limit: int,
) -> str:
    if message.is_hidden or message.is_system:
        return ""

    parts: list[str] = []

    if message.text:
        parts.append(message.text)

    if message.parts:
        for part in message.parts:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if not isinstance(part_text, str) or not part_text:
                continue
            media_type = part.get("mediaType")
            if media_type in {"forward", "quote"}:
                first_line = part_text.split("\n", 1)[0]
                parts.append(first_line[:forward_quote_limit])
            else:
                parts.append(part_text[:regular_part_limit])

    return "\n".join(filter(None, parts))

def prepare_dense_content(message: Message) -> str:
    if message.is_hidden or message.is_system:
        return ""

    parts: list[str] = []

    if message.text:
        parts.append(message.text)

    if message.parts:
        for part in message.parts:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if not isinstance(part_text, str) or not part_text:
                continue
            parts.append(part_text[:600])

    return "\n".join(filter(None, parts))


def prepare_sparse_content(message: Message) -> str:
    return _prepare_base_content(message, forward_quote_limit=80, regular_part_limit=250)


def build_dense_chunk_text(messages: list[Message]) -> str:
    return "\n".join(filter(None, (prepare_dense_content(m) for m in messages)))


def build_sparse_chunk_text(messages: list[Message]) -> str:
    return "\n".join(filter(None, (prepare_sparse_content(m) for m in messages)))


def truncate_content(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.7:
        return truncated[:last_newline]
    return truncated


CHAT_TYPE_RU = {
    "channel": "канал",
    "group": "группа",
    "private": "личный чат",
    "thread": "тред",
}


def chat_header(chat: Chat) -> str:
    """
    Строка-заголовок чанка с контекстом чата.
    Помогает находить чанки по вопросам вида:
      «В какой чат писать про X?»
      «Где обсуждали Y?»
    """
    chat_type_label = CHAT_TYPE_RU.get(chat.type, chat.type)
    return f"[{chat_type_label}: {chat.name}]"


def enrich_sparse_content(
    base_sparse: str,
    chat: Chat,
    mentions: set[str],
    has_forward: bool,
    has_quote: bool,
) -> str:
    """
    Для BM25 добавляем:
    - название чата (точный матч по имени)
    - сигналы о наличии forward/quote для фильтрации
    - упоминания (@-теги, имена) из сообщений
    """
    extra_parts: list[str] = [base_sparse]

    # Название чата
    extra_parts.append(chat.name)

    # Упоминания — часто это имена людей или username'ы
    if mentions:
        extra_parts.append(" ".join(sorted(mentions)))

    # Маркеры для поиска по типам контента
    if has_forward:
        extra_parts.append("forward")
    if has_quote:
        extra_parts.append("quote")

    return "\n".join(filter(None, extra_parts))


def build_chunks(
    chat: Chat,
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    result: list[IndexAPIItem] = []
    header = chat_header(chat)

    def build_text_and_ranges(
        messages: list[Message],
    ) -> tuple[str, list[tuple[int, int, str, Message]]]:
        text_parts: list[str] = []
        message_ranges: list[tuple[int, int, str, Message]] = []
        position = 0

        for index, message in enumerate(messages):
            text = render_message(message)
            if not text:
                continue
            if index > 0 and text_parts:
                text_parts.append("\n")
                position += 1
            start = position
            text_parts.append(text)
            position += len(text)
            message_ranges.append((start, position, message.id, message))

        return "".join(text_parts), message_ranges

    def slice_tail(text: str, tail_size: int) -> str:
        if tail_size <= 0:
            return ""
        return text[max(0, len(text) - tail_size):]

    overlap_text, _ = build_text_and_ranges(overlap_messages)
    previous_chunk_text = slice_tail(overlap_text, OVERLAP_SIZE)

    new_text, new_message_ranges = build_text_and_ranges(new_messages)

    for start in range(0, len(new_text), CHUNK_SIZE):
        chunk_body = new_text[start : start + CHUNK_SIZE]
        if not chunk_body:
            continue

        chunk_body_ranges = [
            (
                max(message_start, start) - start,
                min(message_end, start + len(chunk_body)) - start,
                message_id,
                message,
            )
            for message_start, message_end, message_id, message in new_message_ranges
            if message_end > start and message_start < start + len(chunk_body)
        ]

        chunk_overlap = previous_chunk_text
        chunk_text = chunk_overlap
        if chunk_text and chunk_body:
            chunk_text += "\n"
        chunk_text += chunk_body

        chunk_messages = [message for _, _, _, message in chunk_body_ranges]

        # page_content: заголовок чата + полный текст чанка
        # Реранкер видит этот текст -> заголовок помогает ему понять контекст
        page_content = f"{header}\n{chunk_text}"

        # dense_content: заголовок + семантически богатый текст сообщений
        raw_dense = build_dense_chunk_text(chunk_messages)
        dense_content = f"{header}\n{truncate_content(raw_dense)}" if raw_dense else page_content

        # sparse_content: добавляем chat.name, тип и упоминания -> лексический охват
        raw_sparse = build_sparse_chunk_text(chunk_messages)
        all_mentions: set[str] = set()
        for msg in chunk_messages:
            if msg.mentions:
                all_mentions.update(msg.mentions)

        participants = {msg.sender_id for msg in chunk_messages}
        has_forward = any(msg.is_forward for msg in chunk_messages)
        has_quote = any(msg.is_quote for msg in chunk_messages)
        chunk_timestamps = [msg.time for msg in chunk_messages if msg.time]
        date_start = min(chunk_timestamps) if chunk_timestamps else None
        date_end = max(chunk_timestamps) if chunk_timestamps else None

        sparse_content = enrich_sparse_content(
            truncate_content(raw_sparse) if raw_sparse else chunk_text,
            chat,
            all_mentions,
            has_forward=has_forward,
            has_quote=has_quote,
        )

        result.append(
            IndexAPIItem(
                page_content=page_content,
                dense_content=dense_content,
                sparse_content=sparse_content,
                message_ids=[message_id for _, _, message_id, _ in chunk_body_ranges],
                participants=list(participants),
                mentions=list(all_mentions),
                has_forward=has_forward,
                has_quote=has_quote,
                date_start=date_start,
                date_end=date_end,
            )
        )
        previous_chunk_text = slice_tail(chunk_text, OVERLAP_SIZE)

    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.chat,
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    logger.info(
        "Loading sparse model %s from cache %s",
        SPARSE_MODEL_NAME,
        FASTEMBED_CACHE_PATH,
    )
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors: list[SparseVector] = []
    for item in model.embed(texts):
        vectors.append(
            SparseVector(
                indices=item.indices.tolist(),
                values=item.values.tolist(),
            )
        )
    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()