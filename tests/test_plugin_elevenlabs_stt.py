"""Unit tests for ElevenLabs STT plugin configuration."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict
from yarl import URL

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIStatusError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import stt as elevenlabs_stt
from livekit.plugins.elevenlabs._utils import trace_id_from_headers

pytestmark = pytest.mark.plugin("elevenlabs")


class _EventSink:
    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


def _new_stream(
    *,
    server_vad=NOT_GIVEN,
    language: str | None = "en",
    include_timestamps: bool = False,
    include_language_detection=NOT_GIVEN,
    secondary_languages: list[str] | Any = NOT_GIVEN,
    model_id: str = "scribe_v2_realtime",
    commit_strategy=NOT_GIVEN,
    multilang_mode: bool = False,
    turn_probability_threshold=NOT_GIVEN,
) -> elevenlabs_stt.SpeechStream:
    # mirrors what STT.__init__ stores: both language options are normalized on the way in
    stream = object.__new__(elevenlabs_stt.SpeechStream)
    stream._opts = elevenlabs_stt.STTOptions(
        model_id=model_id,
        api_key="test-key",
        base_url=elevenlabs_stt.API_BASE_URL_V1,
        language_code=None,
        secondary_languages=[LanguageCode(code) for code in secondary_languages]
        if secondary_languages is not NOT_GIVEN
        else NOT_GIVEN,
        include_language_detection=include_language_detection,
        tag_audio_events=True,
        include_timestamps=include_timestamps,
        sample_rate=16000,
        server_vad=server_vad,
        keyterms=NOT_GIVEN,
        no_verbatim=False,
        enable_logging=True,
        previous_text=None,
        commit_strategy=commit_strategy,
        multilang_mode=multilang_mode,
        turn_probability_threshold=turn_probability_threshold,
    )
    stream._language = LanguageCode(language) if language else None
    stream._event_ch = _EventSink()
    stream._speaking = False
    stream._last_partial_text = ""
    stream._final_alternatives = []
    stream._uncommitted_samples = 0
    stream._commit_pending = False
    stream._commit_answered = asyncio.Event()
    stream._commit_reason = None
    stream._commit_probability = None
    stream._commit_text_age = None
    stream._answered_commit_reason = None
    stream._last_turn_probability = None
    stream._partial_text_changed_at = None
    stream._commit_copy_seen = None
    stream._commit_forced = False
    stream._forced_commit_deadline = None
    stream._forced_commit_armed = asyncio.Event()
    stream._input_ch = SimpleNamespace(start_time_offset=0.0)
    return stream


def _committed_transcript(
    text: str,
    *,
    with_timestamps: bool = False,
    language_code: str | None = None,
) -> dict:
    message: dict = {
        "message_type": "committed_transcript_with_timestamps"
        if with_timestamps
        else "committed_transcript",
        "text": text,
        "words": [
            {"text": text, "start": 0.1, "end": 0.4},
        ]
        if text
        else [],
    }
    # the server only carries the detected language on the delayed copy, and only when
    # language detection is enabled
    if language_code is not None:
        message["language_code"] = language_code
    return message


def _partial_transcript(text: str) -> dict:
    return {"message_type": "partial_transcript", "text": text, "words": []}


@pytest.mark.parametrize("server_vad", [None, {"vad_silence_threshold_secs": 0.5}])
@pytest.mark.parametrize("end_time", [0.4, 0.0])
def test_elevenlabs_endpoint_preserves_timing_across_empty_commit(server_vad, end_time):
    stream = _new_stream(server_vad=server_vad, include_timestamps=True)
    stream.start_time_offset = 7.0
    final = _committed_transcript("hello", with_timestamps=True)
    final["words"][0]["end"] = end_time
    stream._process_stream_event(final)
    if server_vad is None:
        stream._process_stream_event(_committed_transcript("", with_timestamps=True))

    endpoint = stream._event_ch.events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []

    stream._process_stream_event(_partial_transcript("next turn"))
    stream._process_stream_event(_committed_transcript("", with_timestamps=True))
    assert stream._event_ch.events[-1].alternatives == []


def _interim_texts(stream: elevenlabs_stt.SpeechStream) -> list[str]:
    return [
        event.alternatives[0].text
        for event in stream._event_ch.events
        if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    ]


def test_untimed_final_invalidates_previous_commit_timing() -> None:
    stream = _new_stream(server_vad=None, include_timestamps=True)
    stream._process_stream_event(_committed_transcript("hello", with_timestamps=True))
    final = _committed_transcript("again", with_timestamps=True)
    final["words"] = []
    stream._process_stream_event(final)
    stream._process_stream_event(_committed_transcript("", with_timestamps=True))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert stream._event_ch.events[-1].alternatives == []


def test_advancing_partial_transcripts_are_forwarded() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_partial_transcript("yeah"))
    stream._process_stream_event(_partial_transcript("yeah please"))

    assert _interim_texts(stream) == ["yeah", "yeah please"]


def test_re_sent_partial_transcript_is_dropped() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    for _ in range(5):
        stream._process_stream_event(_partial_transcript("yeah please"))

    assert _interim_texts(stream) == ["yeah please"]
    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
    ]


def test_the_same_words_after_a_commit_are_forwarded_again() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_partial_transcript("right"))
    stream._process_stream_event(_committed_transcript("right"))
    stream._process_stream_event(_partial_transcript("right"))

    assert _interim_texts(stream) == ["right", "right"]


def test_the_same_words_after_an_empty_commit_are_forwarded_again() -> None:
    stream = _new_stream(server_vad=None)

    stream._process_stream_event(_partial_transcript("right"))
    stream._process_stream_event(_committed_transcript(""))
    stream._process_stream_event(_partial_transcript("right"))

    assert _interim_texts(stream) == ["right", "right"]


def test_server_vad_commit_emits_end_of_speech() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._event_ch.events[1].alternatives[0].text == "hello"
    assert stream._speaking is False


def test_manual_commit_still_waits_for_empty_commit() -> None:
    stream = _new_stream(server_vad=None)

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert stream._speaking is True

    stream._process_stream_event(_committed_transcript(""))

    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert stream._speaking is False


def _stt(**kwargs: object) -> elevenlabs_stt.STT:
    return elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime", **kwargs)


def test_no_verbatim_defaults_to_false() -> None:
    assert _stt()._opts.no_verbatim is False


def test_no_verbatim_can_be_enabled() -> None:
    assert _stt(no_verbatim=True)._opts.no_verbatim is True


def test_update_options_sets_no_verbatim() -> None:
    instance = _stt()
    assert instance._opts.no_verbatim is False
    instance.update_options(no_verbatim=True)
    assert instance._opts.no_verbatim is True


def test_update_options_leaves_no_verbatim_untouched_when_not_given() -> None:
    instance = _stt(no_verbatim=True)
    instance.update_options(tag_audio_events=False)
    assert instance._opts.no_verbatim is True


def test_update_options_forwards_no_verbatim_to_active_streams() -> None:
    # no_verbatim is a WebSocket query param applied at connect time, so a live
    # realtime stream must be told to reconnect. Verify STT.update_options
    # forwards it to active streams (which trigger a reconnect).
    instance = _stt()
    captured: dict[str, object] = {}

    class _FakeStream:
        def update_options(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake = _FakeStream()
    instance._streams.add(fake)
    instance.update_options(no_verbatim=True)
    assert captured.get("no_verbatim") is True


def test_enable_logging_defaults_to_true() -> None:
    assert _stt()._opts.enable_logging is True


def test_enable_logging_can_be_disabled() -> None:
    assert _stt(enable_logging=False)._opts.enable_logging is False


def test_previous_text_is_kept_for_realtime_model() -> None:
    assert _stt(previous_text="prior context")._opts.previous_text == "prior context"


def test_previous_text_is_ignored_for_non_realtime_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key",
            model="scribe_v2",
            previous_text="prior context",
        )

    assert instance._opts.previous_text is None
    assert any("previous_text" in record.message for record in caplog.records)


@pytest.mark.parametrize(("enable_logging", "expected"), [(True, "true"), (False, "false")])
async def test_connect_ws_includes_enable_logging(enable_logging: bool, expected: str) -> None:
    # enable_logging is a WebSocket query param. Verify it is forwarded to the
    # realtime connect URL with the expected lowercase boolean string.
    stream = _new_stream()
    stream._opts.enable_logging = enable_logging

    class _ConnOptions:
        timeout = 5.0

    stream._conn_options = _ConnOptions()

    captured: dict[str, object] = {}

    class _FakeSession:
        async def ws_connect(self, url: str, **kwargs: object) -> object:
            captured["url"] = url
            return object()

    stream._session = _FakeSession()

    await stream._connect_ws()

    assert f"enable_logging={expected}" in captured["url"]


async def _connect_ws_url(stream: elevenlabs_stt.SpeechStream) -> str:
    """Run _connect_ws against a fake session and return the realtime connect URL."""

    class _ConnOptions:
        timeout = 5.0

    stream._conn_options = _ConnOptions()

    captured: dict[str, str] = {}

    class _FakeSession:
        async def ws_connect(self, url: str, **kwargs: object) -> object:
            captured["url"] = url
            return object()

    stream._session = _FakeSession()

    await stream._connect_ws()
    return captured["url"]


async def test_connect_ws_includes_keyterms() -> None:
    # keyterms bias the realtime model and are sent as repeated query params on
    # the connect URL.
    stream = _new_stream()
    stream._opts.keyterms = ["nginx", "Grafana Loki", "Ærø"]

    url = await _connect_ws_url(stream)

    assert "keyterms=nginx" in url
    assert "keyterms=Grafana%20Loki" in url
    assert "keyterms=%C3%86r%C3%B8" in url


async def test_connect_ws_escapes_query_delimiters_in_keyterms() -> None:
    # Keyterms are free-form text. Unescaped, "&" would split the term and inject
    # a bogus query param and "#" would truncate it, so both must be encoded and
    # must survive a round trip through the URL parser aiohttp uses.
    stream = _new_stream()
    stream._opts.keyterms = ["Smith & Sons", "C#"]

    url = await _connect_ws_url(stream)

    assert "keyterms=Smith%20%26%20Sons" in url
    assert "keyterms=C%23" in url
    assert URL(url).query.getall("keyterms") == ["Smith & Sons", "C#"]


async def test_connect_ws_omits_keyterms_when_not_given() -> None:
    url = await _connect_ws_url(_new_stream())

    assert "keyterms=" not in url


def test_update_options_forwards_keyterms_to_active_streams() -> None:
    # keyterms are a WebSocket query param applied at connect time, so a live
    # realtime stream must be told to reconnect. Verify STT.update_options
    # forwards them to active streams (which trigger a reconnect).
    instance = _stt()
    captured: dict[str, object] = {}

    class _FakeStream:
        def update_options(self, **kwargs: object) -> None:
            captured.update(kwargs)

    # _streams is a WeakSet: keep a strong reference so the fake survives.
    fake = _FakeStream()
    instance._streams.add(fake)
    instance.update_options(keyterms=["nginx"])
    assert captured.get("keyterms") == ["nginx"]


def test_stream_update_options_sets_keyterms_and_requests_reconnect() -> None:
    stream = _new_stream()
    stream._reconnect_event = asyncio.Event()

    stream.update_options(keyterms=["nginx"])

    assert stream._opts.keyterms == ["nginx"]
    assert stream._reconnect_event.is_set()


async def test_connect_ws_normalizes_the_primary_language() -> None:
    # LanguageCode keeps the region ("en-US") but the realtime API rejects it, so the primary
    # language goes on the wire through the same normalization the secondary ones get
    stream = _new_stream(language="en_US", secondary_languages=["ru-RU"])

    url = await _connect_ws_url(stream)

    assert URL(url).query.getall("language_code") == ["en"]
    assert URL(url).query.getall("secondary_languages") == ["ru"]


async def test_connect_ws_includes_secondary_languages() -> None:
    # secondary languages ride along with the pinned primary one and are sent as
    # repeated query params, the only serialization the realtime API accepts.
    stream = _new_stream(language="en", secondary_languages=["ru", "es"])

    url = await _connect_ws_url(stream)

    assert "language_code=en" in url
    assert URL(url).query.getall("secondary_languages") == ["ru", "es"]


async def test_connect_ws_normalizes_secondary_languages() -> None:
    # the realtime API takes ISO-639-1/3 and rejects anything else, including the region-tagged
    # tags LanguageCode produces, so names and regions are mapped before the connect URL
    stream = _new_stream(language="en", secondary_languages=["ru_RU", "french", "spa"])

    url = await _connect_ws_url(stream)

    assert URL(url).query.getall("secondary_languages") == ["ru", "fr", "es"]


async def test_connect_ws_omits_secondary_languages_when_not_given() -> None:
    url = await _connect_ws_url(_new_stream(language="en"))

    assert "secondary_languages=" not in url


def test_secondary_languages_are_normalized_on_the_options() -> None:
    instance = elevenlabs_stt.STT(
        api_key="test-key",
        model="scribe_v2_realtime",
        language_code="en_US",
        secondary_languages=["ru_RU", "french", "spa"],
    )

    assert instance._opts.language_code == LanguageCode("en-US")
    assert instance._opts.secondary_languages == ["ru-RU", "fr", "es"]
    assert all(
        isinstance(code, LanguageCode) for code in cast(list, instance._opts.secondary_languages)
    )


def test_secondary_languages_ignored_for_batch_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key", model="scribe_v2", secondary_languages=["ru"]
        )

    assert instance._opts.secondary_languages is NOT_GIVEN
    assert "only supported for realtime models" in caplog.text


async def test_connect_ws_requests_language_detection_when_no_language_is_pinned() -> None:
    url = await _connect_ws_url(_new_stream(language=None))

    assert "include_language_detection=true" in url


async def test_connect_ws_omits_language_detection_when_language_is_pinned() -> None:
    url = await _connect_ws_url(_new_stream(language="en"))

    assert "include_language_detection" not in url


async def test_connect_ws_requests_language_detection_when_explicitly_enabled() -> None:
    stream = _new_stream(language="en", include_language_detection=True)

    url = await _connect_ws_url(stream)

    assert "include_language_detection=true" in url


async def test_connect_ws_omits_language_detection_when_explicitly_disabled() -> None:
    url = await _connect_ws_url(_new_stream(language=None, include_language_detection=False))

    assert "include_language_detection" not in url


def test_final_transcript_reports_the_detected_language() -> None:
    # with detection on, the detected language only reaches the delayed copy of the
    # commit, so that copy has to be the final one or every transcript is labelled
    # with the pinned language (or "en" when nothing is pinned).
    stream = _new_stream(
        server_vad={"vad_silence_threshold_secs": 0.5},
        language="en",
        include_language_detection=True,
        secondary_languages=["ru"],
    )

    stream._process_stream_event(_committed_transcript("привет"))
    stream._process_stream_event(
        _committed_transcript("привет", with_timestamps=True, language_code="ru")
    )

    finals = [
        event
        for event in stream._event_ch.events
        if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "ru"


def test_autodetected_language_reaches_the_final_transcript() -> None:
    # without a pinned language the plugin asks the server to detect one; that language
    # only rides on the delayed copy, so dropping it labelled every transcript "en"
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5}, language=None)

    stream._process_stream_event(
        _committed_transcript("привет", with_timestamps=True, language_code="ru")
    )
    stream._process_stream_event(_committed_transcript("привет"))

    finals = [
        event
        for event in stream._event_ch.events
        if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "ru"
    # reading the timestamped copy must not start handing out word timings the caller
    # never asked for
    assert finals[0].alternatives[0].words is None


def test_final_transcript_keeps_the_plain_copy_without_detection() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5}, language="es")

    stream._process_stream_event(_committed_transcript("hola"))
    stream._process_stream_event(_committed_transcript("hola", with_timestamps=True))

    finals = [
        event
        for event in stream._event_ch.events
        if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "es"


@pytest.mark.parametrize(
    "message_type",
    ["partial_transcript", "committed_transcript", "committed_transcript_with_timestamps"],
)
@pytest.mark.parametrize(
    ("language", "language_data", "expected_language"),
    [
        ("es", {}, "es"),
        ("es", {"language_code": None}, "es"),
        ("es", {"language_code": ""}, "es"),
        ("es", {"language_code": "fra"}, "fr"),
        (None, {}, "en"),
        (None, {"language_code": None}, "en"),
        (None, {"language_code": ""}, "en"),
        (None, {"language_code": "fra"}, "fr"),
    ],
)
def test_transcript_language_falls_back_to_configured_language(
    message_type: str,
    language: str | None,
    language_data: dict[str, str | None],
    expected_language: str,
) -> None:
    stream = _new_stream(
        language=language,
        include_timestamps=message_type == "committed_transcript_with_timestamps",
        include_language_detection=False,
    )

    stream._process_stream_event({"message_type": message_type, "text": "hola", **language_data})

    transcript = stream._event_ch.events[-1]
    assert transcript.type == (
        stt.SpeechEventType.INTERIM_TRANSCRIPT
        if message_type == "partial_transcript"
        else stt.SpeechEventType.FINAL_TRANSCRIPT
    )
    assert transcript.alternatives[0].language == expected_language


class _FakeWS:
    """Records outgoing messages. receive() parks so recv_task stays alive."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._closed = asyncio.Event()
        self.received: asyncio.Queue[dict] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self) -> Any:
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT, data=json.dumps(await self.received.get())
        )

    async def close(self) -> None:
        self._closed.set()


def _live_stream(ws: _FakeWS, **kwargs: Any) -> elevenlabs_stt.SpeechStream:
    """A real SpeechStream running its real _run loop against a fake socket."""
    kwargs.setdefault("model", "scribe_v2_realtime")
    instance = elevenlabs_stt.STT(api_key="test-key", **kwargs)
    opts = dataclasses.replace(instance._opts)
    stream = elevenlabs_stt.SpeechStream(
        stt=instance,
        opts=opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        language=None,
        http_session=cast(Any, SimpleNamespace(closed=False)),
    )

    async def _fake_connect() -> Any:
        return ws

    # patched before the _run task gets its first tick, so no real socket is opened
    stream._connect_ws = _fake_connect  # type: ignore[method-assign]
    return stream


# Scribe rejects a commit with less audio than this since the previous one
_MIN_COMMIT_MS = 300


def _frame(ms: int, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream to send"
        await asyncio.sleep(0.01)


async def test_flush_commits_the_turn_when_no_audio_is_left_to_send() -> None:
    # 300ms is a whole number of repack chunks, so AudioByteStream.flush() returns no frames.
    # The commit must still go out: Scribe v2 does not finalize a turn without one, and
    # gating it on the per-frame loop dropped it here (roughly 1 flush in 5).
    ws = _FakeWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(_MIN_COMMIT_MS))
        await _wait_until(lambda: len(ws.sent) == 6)

        stream.flush()
        await _wait_until(lambda: len(ws.sent) == 7)

        assert [msg["commit"] for msg in ws.sent] == [False] * 6 + [True]
        assert ws.sent[-1]["audio_base_64"] == ""
    finally:
        await stream.aclose()


async def test_flush_commits_after_the_buffered_audio() -> None:
    # a partial chunk is still pending: it has to reach the server before the commit,
    # otherwise the tail of the turn is transcribed against the next one
    ws = _FakeWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(_MIN_COMMIT_MS + 30))
        stream.flush()
        await _wait_until(lambda: len(ws.sent) == 8)

        assert [msg["commit"] for msg in ws.sent] == [False] * 7 + [True]
        assert ws.sent[-2]["audio_base_64"] != ""
    finally:
        await stream.aclose()


def _audio_ms_before_each_commit(sent: list[dict], sample_rate: int = 16000) -> list[float]:
    audio_ms: list[float] = []
    samples = 0
    for msg in sent:
        if msg["commit"]:
            audio_ms.append(samples * 1000 / sample_rate)
            samples = 0
        else:
            samples += len(base64.b64decode(msg["audio_base_64"])) // 2
    return audio_ms


async def test_flush_pads_a_short_turn_with_silence() -> None:
    # Scribe answers a commit with less than 300ms of audio since the previous one with
    # commit_throttled and closes the session, dropping the uncommitted audio. The commit still
    # has to go out on the flush: the caller waits on its final transcript, and no more audio
    # may follow (push-to-talk, session close).
    ws = _FakeWS()
    stream = _live_stream(ws)
    speech = bytes(range(1, 201)) * 16  # 100ms without a zero sample
    try:
        # the first commit on a connection counts the audio since the connect
        stream.push_frame(
            rtc.AudioFrame(
                data=speech, sample_rate=16000, num_channels=1, samples_per_channel=len(speech) // 2
            )
        )
        stream.flush()
        await _wait_until(lambda: sum(msg["commit"] for msg in ws.sent) == 1)

        stream.flush()
        await _wait_until(lambda: sum(msg["commit"] for msg in ws.sent) == 2)

        assert _audio_ms_before_each_commit(ws.sent) == [300, 300]
        silence_ms = (_MIN_COMMIT_MS - 100) + _MIN_COMMIT_MS
        audio = b"".join(base64.b64decode(msg["audio_base_64"]) for msg in ws.sent)
        assert audio == speech + bytes(16000 * 2 * silence_ms // 1000)
    finally:
        await stream.aclose()


def test_trace_id_from_headers() -> None:
    # header lookup is case-insensitive, and an absent header returns None
    assert trace_id_from_headers(CIMultiDict({"X-Trace-Id": "trace-1"})) == "trace-1"
    assert trace_id_from_headers(CIMultiDict()) is None
    assert trace_id_from_headers(None) is None


def test_speech_confidence_from_word_logprobs() -> None:
    # confident transcription: word logprobs near 0 -> confidence near 1 (spacing tokens ignored)
    words = [
        {"type": "word", "logprob": -0.01},
        {"type": "spacing", "logprob": -2.0},
        {"type": "word", "logprob": -0.05},
    ]
    assert 0.9 < elevenlabs_stt._speech_confidence(words) <= 1.0


def test_speech_confidence_flags_low_quality_transcription() -> None:
    # uncertain transcription (very negative logprobs) -> low confidence
    words = [{"type": "word", "logprob": -2.5}, {"type": "word", "logprob": -3.0}]
    assert elevenlabs_stt._speech_confidence(words) < 0.2


def test_speech_confidence_defaults_to_zero_without_logprobs() -> None:
    # no words, or words without logprobs (e.g. non-timestamped commit) -> default 0.0
    assert elevenlabs_stt._speech_confidence(None) == 0.0
    assert elevenlabs_stt._speech_confidence([{"text": "hi", "start": 0.1, "end": 0.4}]) == 0.0


def test_committed_transcript_sets_confidence() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(
        {
            "message_type": "committed_transcript",
            "text": "hello",
            "words": [
                {"text": "hello", "start": 0.1, "end": 0.4, "type": "word", "logprob": -0.02}
            ],
        }
    )

    final = stream._event_ch.events[1]
    assert final.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].confidence > 0.9


@pytest.mark.parametrize(
    ("model", "server_vad", "manual_flush"),
    [
        ("scribe_v2_realtime", NOT_GIVEN, True),
        ("scribe_v2_realtime", None, True),
        ("scribe_v2_realtime", {}, False),
        ("scribe_v2", NOT_GIVEN, False),
    ],
)
def test_manual_flush_capability(model, server_vad, manual_flush) -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model=model, server_vad=server_vad)
    assert instance.capabilities.manual_flush is manual_flush


def test_manual_flush_capability_tracks_server_vad_updates() -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime")
    instance.update_options(server_vad={})
    assert instance.capabilities.manual_flush is False
    instance.update_options(server_vad=None)
    assert instance.capabilities.manual_flush is True


@pytest.mark.parametrize(
    "message_type", ["auth_error", "quota_exceeded", "transcriber_error", "input_error", "error"]
)
def test_provider_error_content_stays_in_pii_attributes(
    message_type: str, caplog: pytest.LogCaptureFixture
) -> None:
    payload = {
        "message_type": message_type,
        "message": "customer transcript",
        "details": "provider credential",
    }
    stream = _new_stream()
    with caplog.at_level(logging.ERROR), pytest.raises(APIConnectionError) as exc:
        stream._process_stream_event(payload)

    assert str(exc.value) == f"ElevenLabs STT error [{message_type}]"
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == f"ElevenLabs STT error [{message_type}]"
    assert record.__dict__["lk.pii.data"] == payload
    assert record.exc_info is None


@pytest.mark.parametrize(
    ("message_type", "retryable"),
    [
        ("auth_error", False),
        ("quota_exceeded", False),
        ("input_error", False),
        ("transcriber_error", True),
        ("error", True),
    ],
)
async def test_provider_error_reaches_stream_consumer(message_type: str, retryable: bool) -> None:
    ws = _FakeWS()
    stream = _live_stream(ws)
    if retryable:
        stream._conn_options = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)
    try:
        ws.received.put_nowait({"message_type": message_type, "message": "customer content"})
        with pytest.raises(
            APIConnectionError, match=rf"ElevenLabs STT error \[{message_type}\]"
        ) as exc:
            await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert exc.value.retryable is retryable
        assert stream._num_retries == 0
        assert ws._closed.is_set()
    finally:
        await stream.aclose()


def test_audio_chunk_duration_defaults_to_50ms() -> None:
    assert _stt()._opts.audio_chunk_duration_ms == 50


@pytest.mark.parametrize("duration", [0, -1, 0.5, 100.0, True, False, None, "100"])
def test_invalid_audio_chunk_duration_is_rejected(duration: Any) -> None:
    with pytest.raises(ValueError, match="audio_chunk_duration_ms must be a positive integer"):
        _stt(audio_chunk_duration_ms=duration)


@pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
@pytest.mark.parametrize("duration", [1, 50, 75, 100, 200])
@pytest.mark.parametrize("tail_ms", [0, 1])
async def test_configured_audio_chunks_preserve_audio_and_commit(
    sample_rate: int, duration: int, tail_ms: int
) -> None:
    ws = _FakeWS()
    stream = _live_stream(ws, sample_rate=sample_rate, audio_chunk_duration_ms=duration)
    # Distinct PCM bytes reveal drops, duplication and reordering across input frames.
    total_samples = sample_rate * (max(duration * 2, _MIN_COMMIT_MS) + tail_ms) // 1000
    audio = bytes(i % 251 for i in range(total_samples * 2))
    input_frame_bytes = sample_rate * 20 // 1000 * 2
    chunk_bytes = sample_rate * duration // 1000 * 2
    expected_chunks = [audio[i : i + chunk_bytes] for i in range(0, len(audio), chunk_bytes)]
    try:
        for offset in range(0, len(audio), input_frame_bytes):
            data = audio[offset : offset + input_frame_bytes]
            stream.push_frame(
                rtc.AudioFrame(
                    data=data,
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(data) // 2,
                )
            )
        # Complete chunks must be released before the caller flushes.
        await _wait_until(lambda: len(ws.sent) >= len(audio) // chunk_bytes)
        stream.flush()
        await _wait_until(lambda: len(ws.sent) >= len(expected_chunks) + 1)

        assert [base64.b64decode(msg["audio_base_64"]) for msg in ws.sent[:-1]] == expected_chunks
        assert [msg["commit"] for msg in ws.sent] == [False] * len(expected_chunks) + [True]
        assert all(msg["sample_rate"] == sample_rate for msg in ws.sent)
        assert ws.sent[-1]["audio_base_64"] == ""
    finally:
        await stream.aclose()


_INVALID_KEY = {"detail": {"code": "invalid_api_key", "message": "Invalid API key"}}


@pytest.mark.parametrize(
    ("body", "message", "parsed"),
    [
        (json.dumps(_INVALID_KEY), "Invalid API key", _INVALID_KEY),
        ("", "Unknown ElevenLabs error", ""),
        ("{not json", "{not json", "{not json"),
    ],
)
async def test_recognize_does_not_retry_a_client_error(
    body: str, message: str, parsed: Any
) -> None:
    requests = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(status=401, text=body, content_type="application/json")

    app = web.Application()
    app.router.add_post("/speech-to-text", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        url = str(server.make_url(""))
        batch_stt = elevenlabs_stt.STT(api_key="bad-key", base_url=url, http_session=session)
        with pytest.raises(APIStatusError) as exc_info:
            await batch_stt.recognize([_frame(100)])

    assert exc_info.value.status_code == 401
    assert exc_info.value.message == message
    assert exc_info.value.body == parsed
    assert requests == 1


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.elevenlabs.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", use_realtime=True, http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(keyterms=["LiveKit"]))
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_recognize_language_override_applies_to_one_call() -> None:
    # StreamAdapter passes its stream's language to every recognize() call, so an
    # override must not leak into the STT's own configuration
    sent_languages: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        form = await request.post()
        sent_languages.append(str(form.get("language_code")))
        return web.json_response({"text": "hello", "words": []})

    app = web.Application()
    app.router.add_post("/speech-to-text", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        url = str(server.make_url(""))
        batch_stt = elevenlabs_stt.STT(
            api_key="test-key", base_url=url, language_code="en", http_session=session
        )
        overridden = await batch_stt.recognize([_frame(100)], language="fr")
        default = await batch_stt.recognize([_frame(100)])

    assert sent_languages == ["fr", "en"]
    assert overridden.alternatives[0].language == "fr"
    assert default.alternatives[0].language == "en"
    assert batch_stt._opts.language_code == "en"


# --- Scribe v3 Live: realtime path, turn prediction, multilingual mode


def _stream_of(instance: elevenlabs_stt.STT) -> elevenlabs_stt.SpeechStream:
    """A stream holding the options a real STT instance resolved, without opening a socket."""
    stream = object.__new__(elevenlabs_stt.SpeechStream)
    stream._opts = instance._opts
    stream._language = instance._opts.language_code
    return stream


async def _realtime_query(**kwargs: Any) -> Any:
    """The query of the realtime connect URL an STT built with these options asks for."""
    kwargs.setdefault("model", "scribe_v2_realtime")
    instance = elevenlabs_stt.STT(api_key="test-key", **kwargs)
    return URL(await _connect_ws_url(_stream_of(instance))).query


@pytest.mark.parametrize("model", ["scribe_v2_realtime", "scribe_v3_live"])
async def test_realtime_models_stream_over_the_same_websocket(model: str) -> None:
    assert elevenlabs_stt.STT(api_key="test-key", model=model).capabilities.streaming is True

    instance = elevenlabs_stt.STT(
        api_key="test-key",
        model=model,
        language_code="en",
        secondary_languages=["es"],
        include_language_detection=True,
        server_vad={"vad_silence_threshold_secs": 0.5},
    )
    url = URL(await _connect_ws_url(_stream_of(instance)))

    assert url.path.endswith("/speech-to-text/realtime")
    assert url.query["model_id"] == model
    assert url.query["commit_strategy"] == "vad"
    assert url.query["vad_silence_threshold_secs"] == "0.5"
    assert url.query.getall("secondary_languages") == ["es"]
    assert url.query["include_language_detection"] == "true"
    assert "multilang_mode" not in url.query


def test_scribe_v3_stays_on_the_batch_api() -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v3")

    assert instance.capabilities.streaming is False
    assert instance.capabilities.manual_flush is False


def test_secondary_language_updates_reach_scribe_v3_live(
    caplog: pytest.LogCaptureFixture,
) -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v3_live", language_code="en")

    with caplog.at_level("WARNING"):
        instance.update_options(secondary_languages=["es"])

    assert instance._opts.secondary_languages == ["es"]
    assert "will be ignored" not in caplog.text


async def test_turn_prediction_is_requested_for_scribe_v3_live() -> None:
    query = await _realtime_query(
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 1.2},
    )

    assert query["commit_strategy"] == "turn_prediction"
    # VAD stays active under turn prediction, so its options still go out
    assert query["vad_silence_threshold_secs"] == "1.2"


async def test_turn_prediction_needs_no_server_vad_options() -> None:
    query = await _realtime_query(model="scribe_v3_live", commit_strategy="turn_prediction")

    assert query["commit_strategy"] == "turn_prediction"
    assert "vad_silence_threshold_secs" not in query


def test_each_commit_ends_the_turn_under_turn_prediction() -> None:
    stream = _new_stream(model_id="scribe_v3_live", commit_strategy="turn_prediction")

    stream._process_stream_event(_partial_transcript("hello"))
    stream._process_stream_event(_committed_transcript("hello there"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._speaking is False


async def test_an_explicit_commit_strategy_overrides_the_server_vad_default() -> None:
    assert (await _realtime_query(commit_strategy="vad"))["commit_strategy"] == "vad"
    assert (await _realtime_query(commit_strategy="manual", server_vad={}))[
        "commit_strategy"
    ] == "manual"


async def test_turn_prediction_is_ignored_for_other_realtime_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        query = await _realtime_query(commit_strategy="turn_prediction")

    assert "`turn_prediction` commit strategy is only supported for Scribe v3 Live" in caplog.text
    assert query["commit_strategy"] == "manual"


def test_the_commit_strategy_is_ignored_for_batch_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key", model="scribe_v3", commit_strategy="turn_prediction"
        )

    assert instance._opts.commit_strategy is NOT_GIVEN
    assert "`commit_strategy` is only supported for realtime models" in caplog.text


def test_an_unknown_commit_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="commit_strategy must be one of"):
        elevenlabs_stt.STT(
            api_key="test-key", model="scribe_v3_live", commit_strategy=cast(Any, "turns")
        )


async def test_multilingual_mode_is_requested_for_scribe_v3_live() -> None:
    query = await _realtime_query(model="scribe_v3_live", multilang_mode=True)

    assert query["multilang_mode"] == "true"


async def test_multilingual_mode_is_ignored_for_other_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        query = await _realtime_query(multilang_mode=True)

    assert "`multilang_mode` is only supported for Scribe v3 Live" in caplog.text
    assert "multilang_mode" not in query


@pytest.mark.parametrize(
    ("kwargs", "manual_flush"),
    [
        ({}, True),
        ({"server_vad": {}}, False),
        ({"commit_strategy": "turn_prediction"}, False),
        ({"commit_strategy": "turn_prediction", "server_vad": {}}, False),
        ({"commit_strategy": "vad"}, False),
        ({"commit_strategy": "manual", "server_vad": {}}, True),
    ],
)
def test_manual_flush_follows_the_commit_strategy(kwargs: dict, manual_flush: bool) -> None:
    # a server that commits on its own must not also get the framework's commits
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v3_live", **kwargs)

    assert instance.capabilities.manual_flush is manual_flush


def test_server_vad_updates_keep_an_explicit_commit_strategy() -> None:
    instance = elevenlabs_stt.STT(
        api_key="test-key", model="scribe_v3_live", commit_strategy="turn_prediction"
    )

    instance.update_options(server_vad={"vad_silence_threshold_secs": 0.8})
    assert instance.capabilities.manual_flush is False
    assert _stream_of(instance)._commit_strategy == "turn_prediction"

    instance.update_options(server_vad=None)
    assert instance.capabilities.manual_flush is False
    assert _stream_of(instance)._commit_strategy == "turn_prediction"


# --- Scribe v3 Live: turn end at a caller-chosen turn probability


def _turn_partial(text: str, probability: float | None) -> dict:
    message: dict = {"message_type": "partial_transcript", "text": text}
    if probability is not None:
        message["turn_probability"] = probability
    return message


def _commits(ws: _FakeWS) -> int:
    return sum(msg["commit"] for msg in ws.sent)


def _threshold_stream(
    ws: _FakeWS, threshold: float = 0.93, **kwargs: Any
) -> elevenlabs_stt.SpeechStream:
    return _live_stream(ws, model="scribe_v3_live", turn_probability_threshold=threshold, **kwargs)


async def _send_audio(stream: elevenlabs_stt.SpeechStream, ws: _FakeWS, ms: int) -> None:
    """Push audio and wait until all of it went out (a whole number of 50ms chunks)."""
    expected = len(ws.sent) + ms // 50
    stream.push_frame(_frame(ms))
    await _wait_until(lambda: len(ws.sent) == expected)


async def _receive(ws: _FakeWS, *messages: dict) -> None:
    """Hand the stream server messages and wait until it has dealt with them."""
    for message in messages:
        ws.received.put_nowait(message)
    await _wait_until(ws.received.empty)
    for _ in range(5):
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("threshold", [0, -0.1, 1.01, 93, True, "0.93", None])
def test_an_invalid_turn_probability_threshold_is_rejected(threshold: Any) -> None:
    with pytest.raises(ValueError, match="turn_probability_threshold must be greater than 0"):
        elevenlabs_stt.STT(
            api_key="test-key", model="scribe_v3_live", turn_probability_threshold=threshold
        )


@pytest.mark.parametrize("threshold", [0.5, 0.93, 1, 1.0])
def test_a_valid_turn_probability_threshold_is_kept(threshold: float) -> None:
    instance = elevenlabs_stt.STT(
        api_key="test-key", model="scribe_v3_live", turn_probability_threshold=threshold
    )

    assert instance._opts.turn_probability_threshold == threshold


async def test_the_threshold_opens_the_session_with_server_vad(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the server's silence commit stays the fallback for a turn that never gets there
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key",
            model="scribe_v3_live",
            turn_probability_threshold=0.93,
            server_vad={"vad_silence_threshold_secs": 0.6},
        )
    query = URL(await _connect_ws_url(_stream_of(instance))).query

    assert query["commit_strategy"] == "vad"
    assert query["vad_silence_threshold_secs"] == "0.6"
    assert "turn_probability_threshold" not in query
    assert instance.capabilities.manual_flush is False
    assert caplog.text == ""


@pytest.mark.parametrize("commit_strategy", ["turn_prediction", "manual"])
async def test_the_threshold_takes_precedence_over_another_commit_strategy(
    commit_strategy: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        query = await _realtime_query(
            model="scribe_v3_live",
            turn_probability_threshold=0.93,
            commit_strategy=commit_strategy,
        )

    assert "`turn_probability_threshold` takes precedence over `commit_strategy`" in caplog.text
    assert query["commit_strategy"] == "vad"


async def test_the_threshold_and_an_explicit_vad_strategy_agree(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        query = await _realtime_query(
            model="scribe_v3_live", turn_probability_threshold=0.93, commit_strategy="vad"
        )

    assert caplog.text == ""
    assert query["commit_strategy"] == "vad"


@pytest.mark.parametrize("model", ["scribe_v2_realtime", "scribe_v3", "scribe_v2"])
def test_the_threshold_is_ignored_for_other_models(
    model: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key", model=model, turn_probability_threshold=0.93
        )

    assert "`turn_probability_threshold` is only supported for Scribe v3 Live" in caplog.text
    assert instance._opts.turn_probability_threshold is NOT_GIVEN
    # the session keeps the strategy it would have had without the option
    assert instance._opts.commit_strategy is NOT_GIVEN
    assert instance.capabilities.manual_flush is (model == "scribe_v2_realtime")


def test_server_vad_updates_keep_the_threshold_s_strategy() -> None:
    instance = elevenlabs_stt.STT(
        api_key="test-key", model="scribe_v3_live", turn_probability_threshold=0.93
    )

    instance.update_options(server_vad=None)

    assert instance.capabilities.manual_flush is False
    assert _stream_of(instance)._commit_strategy == "vad"


async def test_a_partial_at_the_threshold_commits_the_turn_once() -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.93))

        assert _commits(ws) == 1
        assert ws.sent[-1] == {
            "message_type": "input_audio_chunk",
            "audio_base_64": "",
            "commit": True,
            "sample_rate": 16000,
        }

        # partials keep coming, also during silence, until the commit is answered
        await _send_audio(stream, ws, 2 * _MIN_COMMIT_MS)
        await _receive(ws, *[_turn_partial("Ja, genau.", 0.97)] * 3)

        assert _commits(ws) == 1
    finally:
        await stream.aclose()


@pytest.mark.parametrize(
    "partial",
    [
        _turn_partial("Ja, genau.", 0.92),
        _turn_partial("", 0.99),
        _turn_partial("Ja, genau.", None),
        {"message_type": "partial_transcript", "text": "Ja.", "turn_probability": "0.99"},
    ],
)
async def test_a_partial_that_does_not_qualify_commits_nothing(partial: dict) -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, partial)

        assert _commits(ws) == 0
    finally:
        await stream.aclose()


async def test_the_threshold_commit_waits_for_the_minimum_audio() -> None:
    # Scribe answers a commit with less than 300ms of audio behind it with commit_throttled
    # and closes the session
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS - 100)
        await _receive(ws, _turn_partial("Ja.", 0.99))
        assert _commits(ws) == 0

        await _send_audio(stream, ws, 100)
        await _receive(ws, _turn_partial("Ja.", 0.99))
        assert _commits(ws) == 1
        assert _audio_ms_before_each_commit(ws.sent) == [_MIN_COMMIT_MS]
    finally:
        await stream.aclose()


async def test_the_minimum_audio_is_counted_from_the_answer_to_the_last_commit() -> None:
    # Scribe counts the uncommitted audio from where it finished a commit, not from where the
    # client sent it: 350ms sent after a commit was measured as 0.20s against the live API
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.95))
        assert _commits(ws) == 1

        # sent while the commit is unanswered: none of it counts
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _committed_transcript("Ja, genau."), _turn_partial("Und", 0.95))
        assert _commits(ws) == 1

        await _send_audio(stream, ws, _MIN_COMMIT_MS - 100)
        await _receive(ws, _turn_partial("Und dann?", 0.95))
        assert _commits(ws) == 1

        await _send_audio(stream, ws, 100)
        await _receive(ws, _turn_partial("Und dann?", 0.95))
        assert _commits(ws) == 2
    finally:
        await stream.aclose()


async def test_the_server_s_own_commit_starts_the_count_over() -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, 2 * _MIN_COMMIT_MS)
        # the silence fallback committed first; the partial behind it must not commit again
        await _receive(ws, _committed_transcript("Ja, genau."), _turn_partial("Ja", 0.99))

        assert _commits(ws) == 0
    finally:
        await stream.aclose()


async def _events_of(stream: elevenlabs_stt.SpeechStream) -> list[stt.SpeechEventType]:
    events: list[stt.SpeechEventType] = []
    async for event in stream:
        if event.type != stt.SpeechEventType.RECOGNITION_USAGE:
            events.append(event.type)
    return events


async def test_the_threshold_commit_ends_the_turn_once() -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    reader = asyncio.create_task(_events_of(stream))
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.95))
        assert _commits(ws) == 1

        # Scribe sends both copies of a commit; then the empty one of a second commit that
        # crossed ours on the wire, as the server's silence commit does
        await _receive(
            ws,
            _committed_transcript("Ja, genau.", with_timestamps=True),
            _committed_transcript("Ja, genau."),
            _committed_transcript("", with_timestamps=True),
            _committed_transcript(""),
        )
    finally:
        await stream.aclose()

    assert await reader == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_a_second_turn_on_the_same_connection_is_committed_too() -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    reader = asyncio.create_task(_events_of(stream))
    try:
        for text in ("Ja, genau.", "Und dann?"):
            await _send_audio(stream, ws, _MIN_COMMIT_MS)
            await _receive(ws, _turn_partial(text, 0.95))
            await _receive(
                ws,
                _committed_transcript(text, with_timestamps=True),
                _committed_transcript(text),
            )

        assert _commits(ws) == 2
    finally:
        await stream.aclose()

    turn = [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert await reader == turn + turn


async def test_no_threshold_leaves_the_commits_to_the_server() -> None:
    ws = _FakeWS()
    stream = _live_stream(ws, model="scribe_v3_live", commit_strategy="turn_prediction")
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.99))

        assert _commits(ws) == 0
    finally:
        await stream.aclose()


async def test_the_threshold_commit_is_logged_for_tuning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws, threshold=0.93)
    try:
        with caplog.at_level(logging.DEBUG, logger="livekit.plugins.elevenlabs"):
            await _send_audio(stream, ws, _MIN_COMMIT_MS)
            await _receive(ws, _turn_partial("Ja, genau.", 0.96))
    finally:
        await stream.aclose()

    records = [r for r in caplog.records if "committing the turn" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].__dict__["turn_probability"] == 0.96
    assert records[0].__dict__["turn_probability_threshold"] == 0.93


def _turn_ends(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "livekit.plugins.elevenlabs.turns"]


async def test_a_threshold_commit_logs_the_turn_end_at_the_probability_it_was_sent_at(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws, threshold=0.93)
    try:
        with caplog.at_level(logging.INFO, logger="livekit.plugins.elevenlabs.turns"):
            await _send_audio(stream, ws, _MIN_COMMIT_MS)
            await _receive(ws, _turn_partial("Also der Code nutzt ein super", 0.95))
            # partials go on until the commit is answered; the log keeps the one that committed
            await _receive(ws, _turn_partial("Also der Code nutzt ein super", 0.99))
            await _receive(
                ws,
                _committed_transcript("Also der Code nutzt ein super", with_timestamps=True),
                _committed_transcript("Also der Code nutzt ein super"),
            )
    finally:
        await stream.aclose()

    records = _turn_ends(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert records[0].__dict__["ended_by"] == "turn_probability"
    assert records[0].__dict__["turn_probability"] == 0.95
    assert records[0].__dict__["turn_probability_threshold"] == 0.93
    assert records[0].__dict__["text_stable_s"] >= 0


@pytest.mark.parametrize(
    ("commit_strategy", "probability", "ended_by", "threshold"),
    [
        ("turn_prediction", 0.95, "turn_prediction", 0.9),
        ("turn_prediction", 0.4, "silence", 0.9),
        ("vad", 0.4, "silence", None),
    ],
)
def test_the_server_s_own_commit_logs_what_ended_the_turn(
    caplog: pytest.LogCaptureFixture,
    commit_strategy: str,
    probability: float,
    ended_by: str,
    threshold: float | None,
) -> None:
    stream = _new_stream(
        model_id="scribe_v3_live", commit_strategy=commit_strategy, server_vad={}
    )

    with caplog.at_level(logging.INFO, logger="livekit.plugins.elevenlabs.turns"):
        stream._process_stream_event(_turn_partial("Ja, genau.", probability))
        stream._process_stream_event(_committed_transcript("Ja, genau."))

    records = _turn_ends(caplog)
    assert len(records) == 1
    assert records[0].__dict__["ended_by"] == ended_by
    assert records[0].__dict__["turn_probability"] == probability
    assert records[0].__dict__.get("turn_probability_threshold") == threshold


def test_a_forced_commit_logs_no_turn_end_until_the_turn_ends(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _mid_speech_stream()

    with caplog.at_level(logging.INFO, logger="livekit.plugins.elevenlabs.turns"):
        stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))
        assert _turn_ends(caplog) == []
        stream._end_turn_after_forced_commit()

    records = _turn_ends(caplog)
    assert len(records) == 1
    assert records[0].__dict__["ended_by"] == "silence_after_forced_commit"


async def test_a_flush_waits_for_the_answer_to_a_threshold_commit() -> None:
    # a commit sent before the previous one is answered is throttled, and the session closed
    # with it takes the previous commit's transcript along
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.95))
        assert _commits(ws) == 1

        stream.flush()
        await _receive(ws)
        assert _commits(ws) == 1

        await _receive(ws, _committed_transcript("Ja, genau."))
        await _wait_until(lambda: _commits(ws) == 2)
        assert _audio_ms_before_each_commit(ws.sent) == [_MIN_COMMIT_MS, _MIN_COMMIT_MS]
    finally:
        await stream.aclose()


async def test_a_flush_does_not_wait_forever_for_that_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_stt, "_COMMIT_ANSWER_TIMEOUT", 0.05)
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.95))

        stream.flush()
        await _wait_until(lambda: _commits(ws) == 2)
    finally:
        await stream.aclose()


async def test_a_reconnect_forgets_an_unanswered_commit() -> None:
    ws = _FakeWS()
    stream = _threshold_stream(ws)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Ja, genau.", 0.95))
        assert _commits(ws) == 1

        # the answer never comes: the connection is replaced
        stream.update_options(no_verbatim=True)
        await _receive(ws)

        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("Und dann?", 0.95))
        assert _commits(ws) == 2
    finally:
        await stream.aclose()


# --- Scribe v3 Live: the commit it forces after about 35s of audio is not a turn end

_TURN = stt.SpeechEventType
_FORCED_AUDIO_SAMPLES = 35 * 16000


def _mid_speech_stream(**kwargs: Any) -> elevenlabs_stt.SpeechStream:
    """A turn-predicting stream 35s into a segment, the speaker mid-sentence."""
    kwargs.setdefault("commit_strategy", "turn_prediction")
    stream = _new_stream(model_id="scribe_v3_live", **kwargs)
    stream._uncommitted_samples = _FORCED_AUDIO_SAMPLES
    stream._process_stream_event(_turn_partial("bevor sie überhaupt", 0.02))
    return stream


def _event_types(stream: elevenlabs_stt.SpeechStream) -> list[stt.SpeechEventType]:
    return [event.type for event in stream._event_ch.events]


def test_a_forced_commit_keeps_the_turn_open() -> None:
    stream = _mid_speech_stream()

    stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))

    assert _event_types(stream) == [
        _TURN.START_OF_SPEECH,
        _TURN.INTERIM_TRANSCRIPT,
        _TURN.FINAL_TRANSCRIPT,
    ]
    assert stream._speaking is True
    # the next forced commit is another 35s away
    assert stream._uncommitted_samples == 0


def test_the_turn_ends_with_the_segment_after_a_forced_commit() -> None:
    stream = _mid_speech_stream()
    stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))

    stream._process_stream_event(_turn_partial("ein Haus kaufen können.", 0.95))
    stream._process_stream_event(_committed_transcript("ein Haus kaufen können."))

    assert _event_types(stream) == [
        _TURN.START_OF_SPEECH,
        _TURN.INTERIM_TRANSCRIPT,
        _TURN.FINAL_TRANSCRIPT,
        _TURN.INTERIM_TRANSCRIPT,
        _TURN.FINAL_TRANSCRIPT,
        _TURN.END_OF_SPEECH,
    ]
    finals = [e for e in stream._event_ch.events if e.type == _TURN.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == [
        "bevor sie überhaupt",
        "ein Haus kaufen können.",
    ]
    assert stream._speaking is False


def _not_enough_audio(stream: elevenlabs_stt.SpeechStream) -> None:
    stream._uncommitted_samples = 33 * 16000


def _predicted_turn_end(stream: elevenlabs_stt.SpeechStream) -> None:
    stream._process_stream_event(_turn_partial("bevor sie überhaupt", 0.9))


def _no_turn_probability(stream: elevenlabs_stt.SpeechStream) -> None:
    stream._process_stream_event(_turn_partial("bevor sie überhaupt", None))


def _silence_before_the_commit(stream: elevenlabs_stt.SpeechStream) -> None:
    assert stream._partial_text_changed_at is not None
    stream._partial_text_changed_at -= 2.5


def _answer_to_our_commit(stream: elevenlabs_stt.SpeechStream) -> None:
    stream._commit_pending = True


@pytest.mark.parametrize(
    "not_forced",
    [
        _not_enough_audio,
        _predicted_turn_end,
        _no_turn_probability,
        _silence_before_the_commit,
        _answer_to_our_commit,
    ],
)
def test_a_commit_that_is_not_forced_still_ends_the_turn(
    not_forced: Callable[[elevenlabs_stt.SpeechStream], None],
) -> None:
    stream = _mid_speech_stream()
    not_forced(stream)

    stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))

    assert _event_types(stream)[-2:] == [_TURN.FINAL_TRANSCRIPT, _TURN.END_OF_SPEECH]
    assert stream._speaking is False
    assert stream._forced_commit_deadline is None


@pytest.mark.parametrize(
    ("threshold", "probability", "forced"),
    [(0.95, 0.92, True), (0.95, 0.95, False), (0.85, 0.88, False), (0.85, 0.5, True)],
)
def test_the_caller_s_threshold_decides_what_a_turn_end_is(
    threshold: float, probability: float, forced: bool
) -> None:
    stream = _mid_speech_stream(commit_strategy="vad", turn_probability_threshold=threshold)
    stream._process_stream_event(_turn_partial("bevor sie überhaupt", probability))

    stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))

    assert stream._speaking is forced


@pytest.mark.parametrize("timestamps_first", [True, False])
def test_both_copies_of_a_forced_commit_get_the_same_verdict(timestamps_first: bool) -> None:
    # with the language detected the final copy is the one with timestamps; whichever copy
    # comes first starts the count of audio over, so the other cannot be judged by it
    stream = _mid_speech_stream(language=None)
    copies = [
        _committed_transcript("bevor sie überhaupt", with_timestamps=True, language_code="de"),
        _committed_transcript("bevor sie überhaupt"),
    ]

    for copy in copies if timestamps_first else reversed(copies):
        stream._process_stream_event(copy)

    assert _event_types(stream) == [
        _TURN.START_OF_SPEECH,
        _TURN.INTERIM_TRANSCRIPT,
        _TURN.FINAL_TRANSCRIPT,
    ]
    assert stream._speaking is True

    # and the commit after it is judged afresh
    stream._process_stream_event(_turn_partial("ein Haus kaufen können.", 0.95))
    for copy in (
        _committed_transcript("ein Haus kaufen können.", with_timestamps=True),
        _committed_transcript("ein Haus kaufen können."),
    ):
        stream._process_stream_event(copy)

    assert _event_types(stream)[-2:] == [_TURN.FINAL_TRANSCRIPT, _TURN.END_OF_SPEECH]


def test_a_forced_commit_changes_nothing_for_manual_commits() -> None:
    stream = _mid_speech_stream(commit_strategy="manual")

    stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))

    assert _event_types(stream)[-1] == _TURN.FINAL_TRANSCRIPT
    assert stream._speaking is True
    # the caller's own flush ends a manual turn, nothing is left to watch for
    assert stream._forced_commit_deadline is None


def test_a_forced_commit_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    stream = _mid_speech_stream()

    with caplog.at_level(logging.DEBUG, logger="livekit.plugins.elevenlabs"):
        stream._process_stream_event(_committed_transcript("bevor sie überhaupt"))

    records = [r for r in caplog.records if "forced a commit" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].__dict__["uncommitted_audio"] == 35.0
    assert records[0].__dict__["turn_probability"] == 0.02


async def _forced_commit_on(stream: elevenlabs_stt.SpeechStream, ws: _FakeWS) -> None:
    """Bring a live stream to a forced commit: 35s into a segment, mid-sentence."""
    await _send_audio(stream, ws, _MIN_COMMIT_MS)
    stream._uncommitted_samples = _FORCED_AUDIO_SAMPLES
    await _receive(
        ws,
        _turn_partial("bevor sie überhaupt", 0.02),
        _committed_transcript("bevor sie überhaupt", with_timestamps=True),
        _committed_transcript("bevor sie überhaupt"),
    )


def _watch(stream: elevenlabs_stt.SpeechStream) -> tuple[list[Any], asyncio.Task[None]]:
    events: list[Any] = []

    async def _read() -> None:
        async for event in stream:
            if event.type != _TURN.RECOGNITION_USAGE:
                events.append(event.type)

    return events, asyncio.create_task(_read())


async def test_a_turn_left_open_by_a_forced_commit_ends_when_no_speech_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Scribe sends nothing after a forced commit when the speaker stopped right at it: no
    # silence commit, no empty one. Without this the turn would stay open for good.
    monkeypatch.setattr(elevenlabs_stt, "_FORCED_COMMIT_MIN_WAIT", 0.05)
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 0.05},
    )
    events, reader = _watch(stream)
    try:
        await _forced_commit_on(stream, ws)
        await _wait_until(lambda: len(events) == 4)

        assert events == [
            _TURN.START_OF_SPEECH,
            _TURN.INTERIM_TRANSCRIPT,
            _TURN.FINAL_TRANSCRIPT,
            _TURN.END_OF_SPEECH,
        ]
        assert stream._speaking is False
    finally:
        await stream.aclose()
        reader.cancel()


async def test_the_silence_threshold_is_waited_for_before_ending_that_turn() -> None:
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 2.5},
    )
    events, reader = _watch(stream)
    try:
        before = time.monotonic()
        await _forced_commit_on(stream, ws)

        assert stream._forced_commit_deadline is not None
        assert 2.3 < stream._forced_commit_deadline - before < 2.8
        assert events[-1] == _TURN.FINAL_TRANSCRIPT
    finally:
        await stream.aclose()
        reader.cancel()


async def test_speech_after_a_forced_commit_leaves_the_turn_to_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_stt, "_FORCED_COMMIT_MIN_WAIT", 0.05)
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 0.05},
    )
    events, reader = _watch(stream)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        stream._uncommitted_samples = _FORCED_AUDIO_SAMPLES
        # the next segment's first partial arrives with the commit, before the wait is over
        await _receive(
            ws,
            _turn_partial("bevor sie überhaupt", 0.02),
            _committed_transcript("bevor sie überhaupt", with_timestamps=True),
            _committed_transcript("bevor sie überhaupt"),
            _turn_partial("ein Haus", 0.01),
        )
        await asyncio.sleep(0.15)

        assert stream._forced_commit_deadline is None
        assert events == [
            _TURN.START_OF_SPEECH,
            _TURN.INTERIM_TRANSCRIPT,
            _TURN.FINAL_TRANSCRIPT,
            _TURN.INTERIM_TRANSCRIPT,
        ]

        await _receive(
            ws,
            _turn_partial("ein Haus kaufen können.", 0.95),
            _committed_transcript("ein Haus kaufen können.", with_timestamps=True),
            _committed_transcript("ein Haus kaufen können."),
        )
        await _wait_until(lambda: events[-1] == _TURN.END_OF_SPEECH)
        assert events.count(_TURN.END_OF_SPEECH) == 1
        assert events.count(_TURN.FINAL_TRANSCRIPT) == 2
    finally:
        await stream.aclose()
        reader.cancel()


async def test_a_reconnect_does_not_leave_that_turn_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_stt, "_FORCED_COMMIT_MIN_WAIT", 0.3)
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 0.3},
    )
    events, reader = _watch(stream)
    try:
        await _forced_commit_on(stream, ws)
        assert events[-1] == _TURN.FINAL_TRANSCRIPT

        stream.update_options(no_verbatim=True)

        await _wait_until(lambda: events[-1] == _TURN.END_OF_SPEECH)
    finally:
        await stream.aclose()
        reader.cancel()


async def test_the_answer_to_a_flush_is_not_taken_for_a_forced_commit() -> None:
    ws = _FakeWS()
    stream = _live_stream(ws, model="scribe_v3_live", commit_strategy="turn_prediction")
    events, reader = _watch(stream)
    try:
        await _send_audio(stream, ws, _MIN_COMMIT_MS)
        await _receive(ws, _turn_partial("bevor sie überhaupt", 0.02))
        stream.flush()
        await _wait_until(lambda: _commits(ws) == 1)
        # as much audio as a forced commit has behind it, but the commit was ours
        stream._uncommitted_samples = _FORCED_AUDIO_SAMPLES
        await _receive(
            ws,
            _committed_transcript("bevor sie überhaupt", with_timestamps=True),
            _committed_transcript("bevor sie überhaupt"),
        )

        assert events[-2:] == [_TURN.FINAL_TRANSCRIPT, _TURN.END_OF_SPEECH]
    finally:
        await stream.aclose()
        reader.cancel()
