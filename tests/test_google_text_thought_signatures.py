"""Thought signatures of text answers in the Google LLM plugin (`text_thought_signatures`)."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

import pytest
from google.genai import types

from livekit.agents import APIConnectOptions, APIStatusError, llm as agents_llm
from livekit.plugins.google.llm import LLM

pytestmark = pytest.mark.unit

HYDROGEN = "gemini-4-hydrogen"
SIGNATURE = b"\x00\x01signature\xff"
STORED = base64.b64encode(SIGNATURE).decode()

NO_RETRY = APIConnectOptions(max_retry=0, timeout=5)


def _llm(model: str = HYDROGEN, **kwargs: Any) -> LLM:
    return LLM(model=model, api_key="test-key", **kwargs)


def _response(
    *parts: types.Part, finish_reason: types.FinishReason | None = None
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(parts=list(parts)), finish_reason=finish_reason)
        ]
    )


class _FakeStream:
    """Serves one scripted stream and records the contents each request carried."""

    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self._responses = responses
        self.contents: list[list[types.Content]] = []

    async def __call__(self, **kwargs: Any) -> AsyncIterator[types.GenerateContentResponse]:
        self.contents.append(kwargs["contents"])

        async def _iter() -> AsyncIterator[types.GenerateContentResponse]:
            for response in self._responses:
                yield response

        return _iter()


async def _chunks(
    model: LLM,
    responses: list[types.GenerateContentResponse],
    chat_ctx: agents_llm.ChatContext | None = None,
) -> tuple[list[agents_llm.ChatChunk], _FakeStream]:
    fake = _FakeStream(responses)
    model._client.aio.models.generate_content_stream = fake  # type: ignore[method-assign]
    chunks: list[agents_llm.ChatChunk] = []
    async with model.chat(
        chat_ctx=chat_ctx or agents_llm.ChatContext.empty(), conn_options=NO_RETRY
    ) as stream:
        async for chunk in stream:
            chunks.append(chunk)
    return chunks, fake


def _history(**assistant_kwargs: Any) -> agents_llm.ChatContext:
    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="first question")
    chat_ctx.add_message(role="assistant", content="first answer", **assistant_kwargs)
    chat_ctx.add_message(role="user", content="second question")
    return chat_ctx


def _stored(model: str = HYDROGEN) -> dict[str, Any]:
    return {"google": {"thought_signature": STORED, "model": model}}


def _signatures(turns: list[dict]) -> list[bytes | None]:
    return [part.get("thought_signature") for turn in turns for part in turn["parts"]]


class TestSignatureOnTheChunk:
    async def test_a_text_part_carries_its_signature(self) -> None:
        chunks, _ = await _chunks(
            _llm(text_thought_signatures=True),
            [
                _response(
                    types.Part(text="hello", thought_signature=SIGNATURE),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
        )
        deltas = [c.delta for c in chunks if c.delta]
        assert [d.content for d in deltas] == ["hello"]
        assert deltas[0].extra == {"google": {"thought_signature": STORED, "model": HYDROGEN}}

    async def test_the_signature_is_dropped_when_the_option_is_off(self) -> None:
        chunks, _ = await _chunks(
            _llm(),
            [
                _response(
                    types.Part(text="hello", thought_signature=SIGNATURE),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
        )
        deltas = [c.delta for c in chunks if c.delta]
        assert [d.content for d in deltas] == ["hello"]
        assert deltas[0].extra is None

    async def test_a_signature_without_text_still_reaches_the_reader(self) -> None:
        # streaming can deliver the signature on a closing part that has no text
        chunks, _ = await _chunks(
            _llm(text_thought_signatures=True),
            [
                _response(types.Part(text="hello")),
                _response(
                    types.Part(text="", thought_signature=SIGNATURE),
                    finish_reason=types.FinishReason.STOP,
                ),
            ],
        )
        deltas = [c.delta for c in chunks if c.delta]
        assert [d.content for d in deltas] == ["hello", None]
        assert deltas[0].extra is None
        assert deltas[1].extra == _stored()

    async def test_a_thought_summary_keeps_its_signature_but_not_its_text(self) -> None:
        chunks, _ = await _chunks(
            _llm(text_thought_signatures=True),
            [
                _response(types.Part(text="reasoning", thought=True, thought_signature=SIGNATURE)),
                _response(types.Part(text="hello"), finish_reason=types.FinishReason.STOP),
            ],
        )
        deltas = [c.delta for c in chunks if c.delta]
        assert [d.content for d in deltas] == [None, "hello"]
        assert deltas[0].extra == _stored()

    async def test_a_signature_alone_is_not_a_response(self) -> None:
        with pytest.raises(APIStatusError, match="no response generated"):
            await _chunks(
                _llm(text_thought_signatures=True),
                [
                    _response(
                        types.Part(text="", thought_signature=SIGNATURE),
                        finish_reason=types.FinishReason.STOP,
                    )
                ],
            )


class TestSignatureInTheHistory:
    def test_it_goes_back_on_the_answer_it_came_with(self) -> None:
        turns, _ = _history(extra=_stored()).to_provider_format(
            format="google", text_signature_model=HYDROGEN
        )
        assert [t["role"] for t in turns] == ["user", "model", "user"]
        assert _signatures(turns) == [None, SIGNATURE, None]

    def test_it_stays_out_by_default(self) -> None:
        turns, _ = _history(extra=_stored()).to_provider_format(format="google")
        assert _signatures(turns) == [None, None, None]

    def test_another_model_s_signature_is_not_sent(self) -> None:
        turns, _ = _history(extra=_stored("gemini-4-argon")).to_provider_format(
            format="google", text_signature_model=HYDROGEN
        )
        assert _signatures(turns) == [None, None, None]

    def test_an_interrupted_answer_loses_its_signature(self) -> None:
        turns, _ = _history(extra=_stored(), interrupted=True).to_provider_format(
            format="google", text_signature_model=HYDROGEN
        )
        assert _signatures(turns) == [None, None, None]

    def test_an_answer_without_one_is_sent_as_plain_text(self) -> None:
        turns, _ = _history().to_provider_format(format="google", text_signature_model=HYDROGEN)
        assert _signatures(turns) == [None, None, None]

    def test_only_the_last_text_part_of_the_answer_is_signed(self) -> None:
        chat_ctx = agents_llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="question")
        chat_ctx.add_message(role="assistant", content=["one", "two"], extra=_stored())
        turns, _ = chat_ctx.to_provider_format(format="google", text_signature_model=HYDROGEN)
        assert _signatures(turns) == [None, None, SIGNATURE]

    def test_a_user_message_is_never_signed(self) -> None:
        chat_ctx = agents_llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="question", extra=_stored())
        turns, _ = chat_ctx.to_provider_format(format="google", text_signature_model=HYDROGEN)
        assert _signatures(turns) == [None]

    def test_the_stored_form_survives_a_json_round_trip(self) -> None:
        restored = agents_llm.ChatContext.from_dict(_history(extra=_stored()).to_dict())
        turns, _ = restored.to_provider_format(format="google", text_signature_model=HYDROGEN)
        assert _signatures(turns) == [None, SIGNATURE, None]


class TestSignatureOnTheRequest:
    async def test_the_request_carries_the_stored_signature(self) -> None:
        _, fake = await _chunks(
            _llm(text_thought_signatures=True),
            [_response(types.Part(text="hi"), finish_reason=types.FinishReason.STOP)],
            _history(extra=_stored()),
        )
        sent = [part.thought_signature for content in fake.contents[0] for part in content.parts]
        assert sent == [None, SIGNATURE, None]

    async def test_the_request_is_plain_when_the_option_is_off(self) -> None:
        _, fake = await _chunks(
            _llm(),
            [_response(types.Part(text="hi"), finish_reason=types.FinishReason.STOP)],
            _history(extra=_stored()),
        )
        sent = [part.thought_signature for content in fake.contents[0] for part in content.parts]
        assert sent == [None, None, None]
