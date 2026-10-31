import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.llm.errors import LLMConfigurationError, LLMUnavailable
from app.llm.factory import create_provider
from app.llm.reliable import AsyncReliableLLM
from app.llm.types import LLMRequest, Message
from app.schemas import ChatRequest

app = FastAPI(title="Examate RAG starter", version="0.1.0")

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_llm(settings: SettingsDep) -> AsyncReliableLLM:
    return create_provider(settings)


LLMDep = Annotated[AsyncReliableLLM, Depends(get_llm)]

@app.get("/hello")
def say_hello():
    return "hello"

@app.get("/")
def say_root():
    return "You are on root page"

@app.get("/health")
def health(settings: SettingsDep) -> dict:
    """Liveness: the process is up. Must be fast and must not call the LLM or the DB."""
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


def sse(data: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest, provider: LLMDep) -> StreamingResponse:
    llm_request = LLMRequest(
        messages=tuple(Message(role=message.role, content=message.content) for message in request.messages),
        model=request.model,
    )

    async def event_stream() -> AsyncIterator[str]:
        usage = None
        try:
            async for chunk in provider.stream(llm_request):
                if chunk.text:
                    yield sse({"delta": chunk.text})
                if chunk.usage is not None:
                    usage = chunk.usage
        except (LLMUnavailable, LLMConfigurationError) as error:
            yield sse({"error": str(error)}, event="error")
        usage_data = usage or {"input_tokens": 0, "output_tokens": 0, "estimated": False}
        if hasattr(usage_data, "__dict__"):
            usage_data = usage_data.__dict__
        yield sse({"usage": usage_data, "cost_usd": 0.0}, event="done")

    return StreamingResponse(event_stream(), media_type="text/event-stream")    