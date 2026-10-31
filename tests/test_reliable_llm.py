import pytest

from app.llm.errors import LLMConfigurationError, LLMUnavailable
from app.llm.fake import FakeProvider
from app.llm.reliable import ReliableLLM
from app.llm.types import LLMRequest, Message, ProviderError, TextChunk

REQUEST = LLMRequest(
    messages=(Message(role="user", content="Say hello"),),
    model="fake-model",
)


class SequenceProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def stream(self, request):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        yield from outcome


def test_retries_transient_errors_then_succeeds():
    provider = SequenceProvider(
        [
            ProviderError("busy", status_code=503),
            ProviderError("busy", status_code=503),
            iter(FakeProvider(chunks=("ok",)).stream(REQUEST)),
        ]
    )
    delays = []

    chunks = list(ReliableLLM(provider, sleep=delays.append).stream(REQUEST))

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert provider.calls == 3
    assert len(delays) == 2
    assert all(
        0 <= delay <= limit for delay, limit in zip(delays, (0.5, 1.0), strict=True)
    )


def test_does_not_retry_authentication_errors():
    provider = SequenceProvider([ProviderError("bad key", status_code=401)])

    with pytest.raises(LLMConfigurationError):
        list(ReliableLLM(provider, sleep=lambda _: pytest.fail("must not sleep")).stream(REQUEST))

    assert provider.calls == 1


def test_fallback_runs_after_primary_is_unavailable():
    primary = SequenceProvider([ProviderError("down", status_code=503)])
    fallback = FakeProvider(chunks=("fallback",))

    chunks = list(
        ReliableLLM(primary, fallback=fallback, sleep=lambda _: None).stream(REQUEST)
    )

    assert "".join(chunk.text for chunk in chunks) == "fallback"
    assert primary.calls == 4
    assert fallback.calls == 1


def test_partial_output_is_not_retried_or_fallen_back():
    class PartialFailure:
        calls = 0

        def stream(self, request):
            self.calls += 1
            yield TextChunk("partial")
            raise ProviderError("connection lost")

    primary = PartialFailure()
    fallback = FakeProvider(chunks=("fallback",))

    with pytest.raises(LLMUnavailable):
        list(ReliableLLM(primary, fallback=fallback, sleep=lambda _: None).stream(REQUEST))

    assert primary.calls == 1
    assert fallback.calls == 0