import json

import httpx
import pytest

from app.llm.openai_compat import OpenAICompatProvider
from app.llm.types import LLMRequest, Message

REQUEST = LLMRequest(
    messages=(Message(role="user", content="Hello"),),
    model="test-model",
)


@pytest.mark.anyio
async def test_openai_compatible_provider_reads_sse_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        payload = {
            "choices": [{"delta": {"content": "Hi"}}],
        }
        usage = {
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "choices": [],
        }
        body = (
            f"data: {json.dumps(payload)}\n\n"
            f"data: {json.dumps(usage)}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    provider = OpenAICompatProvider(
        "http://test/v1",
        "secret",
        "provider-model",
        transport=httpx.MockTransport(handler),
    )

    chunks = [chunk async for chunk in provider.stream_async(REQUEST)]

    assert [chunk.text for chunk in chunks] == ["Hi", ""]
    assert chunks[-1].usage.input_tokens == 2


@pytest.mark.anyio
async def test_openai_compatible_provider_preserves_status_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    provider = OpenAICompatProvider(
        "http://test/v1",
        "secret",
        "provider-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(Exception) as caught:
        _ = [chunk async for chunk in provider.stream_async(REQUEST)]

    assert caught.value.status_code == 401