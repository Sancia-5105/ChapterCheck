import pytest

from app.llm.errors import LLMUnavailable
from app.llm.fake import FakeProvider
from app.llm.types import LLMRequest, Message, ProviderError, Usage


def request() -> LLMRequest:
    return LLMRequest(
        messages=(Message(role="user", content="Say hello"),),
        model="fake-model",
    )


def test_fake_provider_streams_deterministic_text_and_usage():
    provider = FakeProvider(
        chunks=("one", " two"),
        usage=Usage(input_tokens=3, output_tokens=2),
    )

    chunks = list(provider.stream(request()))

    assert [chunk.text for chunk in chunks] == ["one", " two", ""]
    assert chunks[-1].usage == Usage(input_tokens=3, output_tokens=2)
    assert provider.calls == 1


def test_fake_provider_can_simulate_provider_failure():
    provider = FakeProvider(error=ProviderError("provider is down", status_code=503))

    with pytest.raises(ProviderError, match="provider is down"):
        list(provider.stream(request()))


def test_llm_unavailable_is_distinct_from_provider_errors():
    error = LLMUnavailable("primary provider exhausted retries")

    assert str(error) == "primary provider exhausted retries"