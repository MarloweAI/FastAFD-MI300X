from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None
_STOP_BACKEND = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    return_token_ids: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False

    # OpenAI's chat route spells the cap `max_completion_tokens`; the legacy
    # completions route spells it `max_tokens`. Pydantic drops unknown fields
    # silently, so a client sending only the former would otherwise fall back to
    # `max_tokens`'s default of 16 with no error at all -- a wrong answer, not a
    # failure. Accept both and reconcile in the validator below.
    max_completion_tokens: int | None = None

    # `{"include_usage": true}` asks for a final usage-only chunk. Benchmark
    # clients (vLLM's / InferenceX's `benchmark_serving.py`) use it to read exact
    # completion token counts instead of re-tokenizing the text.
    stream_options: Dict[str, bool] | None = None

    @model_validator(mode="after")
    def _reconcile_max_tokens(self) -> "OpenAICompletionRequest":
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self

    @property
    def want_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


class OpenAIBatchCompletionRequest(BaseModel):
    """MiniSGL extension: submit many homogeneous chat/completion requests at once."""

    model: str

    prompts: List[str] | None = None
    messages_batch: List[List[Message]] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    return_token_ids: bool = False

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    def new_users(self, count: int) -> list[int]:
        return [self.new_user() for _ in range(count)]

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int, return_token_ids: bool = False):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [
                    {
                        "delta": delta,
                        "index": 0,
                        "finish_reason": None,
                        "token_ids": [ack.next_token] if return_token_ids and ack.next_token is not None else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_text_completions(self, uid: int, include_usage: bool = False):
        """SSE for the legacy `/v1/completions` shape: `choices[0].text` deltas.

        Deliberately emits **exactly one `choices` chunk per generated token**. OpenAI
        completions clients derive TTFT from the first such chunk and one inter-token
        latency from every subsequent one (see InferenceX
        `utils/bench_serving/backend_request_func.py`), so a trailing empty-text chunk
        would silently add a spurious ~0 ms ITL sample and drag the median down. The
        terminating `finish_reason` therefore rides on the last *token* chunk rather
        than a chunk of its own, and the optional usage summary carries no `choices`
        key at all -- which is exactly how those clients tell the two apart.
        """
        n_tokens = 0
        pending: dict | None = None

        async for ack in self.wait_for_ack(uid):
            if ack.next_token is None and not ack.incremental_output:
                # Terminator carrying no token: fold it into the previous chunk
                # rather than emitting an extra one.
                if ack.finished:
                    break
                continue

            if pending is not None:
                yield f"data: {json.dumps(pending)}\n\n".encode()

            n_tokens += 1
            pending = {
                "id": f"cmpl-{uid}",
                "object": "text_completion",
                "choices": [
                    {
                        "text": ack.incremental_output or "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            if ack.finished:
                break

        if pending is not None:
            pending["choices"][0]["finish_reason"] = "length"
            yield f"data: {json.dumps(pending)}\n\n".encode()

        if include_usage:
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": f"cmpl-{uid}",
                        "object": "text_completion",
                        "choices": [],
                        "usage": {"completion_tokens": n_tokens},
                    }
                )
                + "\n\n"
            ).encode()

        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming text completion for user %s", uid)

    async def collect_chat_completion(self, uid: int, return_token_ids: bool = False) -> dict:
        text_parts: list[str] = []
        token_ids: list[int] = []
        async for ack in self.wait_for_ack(uid):
            if ack.incremental_output:
                text_parts.append(ack.incremental_output)
            if return_token_ids and ack.next_token is not None:
                token_ids.append(int(ack.next_token))
            if ack.finished:
                break
        return {
            "message": {"role": "assistant", "content": "".join(text_parts)},
            "token_ids": token_ids if return_token_ids else None,
            "finish_reason": "stop",
        }

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE, _STOP_BACKEND
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()
        _GLOBAL_STATE = None
    if _STOP_BACKEND is not None:
        _STOP_BACKEND()
        _STOP_BACKEND = None


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Readiness probe. Registered only after the backend is up (see `lifespan`),
    so a 200 here means the engine can accept work -- which is the contract
    vLLM/SGLang serving harnesses poll for before starting a benchmark."""
    return {"status": "ok"}


@app.post("/v1/completions")
async def v1_text_completions(req: OpenAICompletionRequest, request: Request):
    """OpenAI legacy completions: raw `prompt` in, `choices[0].text` deltas out.

    Distinct from `/v1/chat/completions` in that no chat template is applied, so the
    prompt is tokenized exactly as supplied. That is what makes token accounting
    comparable against vLLM's own `/v1/completions` numbers -- a chat template would
    add template tokens the client did not ask for and shift the measured ISL.
    """
    state = get_global_state()
    assert req.prompt is not None, "'prompt' is required for /v1/completions"

    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(
            state.stream_text_completions(uid, include_usage=req.want_usage),
            request,
            uid,
        ),
        media_type="text/event-stream",
    )


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(
            state.stream_chat_completions(uid, return_token_ids=req.return_token_ids),
            request,
            uid,
        ),
        media_type="text/event-stream",
    )


@app.post("/v1/chat/completions/batch")
async def v1_batch_completions(req: OpenAIBatchCompletionRequest):
    state = get_global_state()
    if req.prompts is not None:
        prompts: list[str | list[dict[str, str]]] = list(req.prompts)
    else:
        assert req.messages_batch is not None, "Either 'prompts' or 'messages_batch' must be provided"
        prompts = [[msg.model_dump() for msg in messages] for messages in req.messages_batch]
    if not prompts:
        return {
            "id": "batchcmpl-empty",
            "object": "chat.completion.batch",
            "choices": [],
        }

    def make_sampling_params() -> SamplingParams:
        return SamplingParams(
            ignore_eos=req.ignore_eos,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

    uids = state.new_users(len(prompts))
    await state.send_one(
        BatchTokenizerMsg(
            data=[
                TokenizeMsg(
                    uid=uid,
                    text=prompt,
                    sampling_params=make_sampling_params(),
                )
                for uid, prompt in zip(uids, prompts, strict=True)
            ]
        )
    )

    results = await asyncio.gather(
        *[
            state.collect_chat_completion(uid, return_token_ids=req.return_token_ids)
            for uid in uids
        ]
    )
    choices = [
        {
            "index": index,
            **result,
        }
        for index, result in enumerate(results)
    ]
    return {
        "id": f"batchcmpl-{uids[0]}-{uids[-1]}",
        "object": "chat.completion.batch",
        "choices": choices,
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


async def shell():
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
    except ImportError as exc:
        raise RuntimeError(
            "Shell mode requires the optional 'shell' dependencies. "
            "Install prompt_toolkit or use `pip install -e .[shell]`."
        ) from exc

    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            need_stop = False
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                if need_stop:
                    break
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(
    config: ServerArgs,
    start_backend: Callable[[], None],
    stop_backend: Callable[[], None] | None = None,
    run_shell: bool = False,
) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE, _STOP_BACKEND

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _STOP_BACKEND = stop_backend
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    try:
        if not run_shell:
            uvicorn.run(app, host=host, port=port)
        else:
            asyncio.run(shell())
    finally:
        if _GLOBAL_STATE is not None:
            _GLOBAL_STATE.shutdown()
            _GLOBAL_STATE = None
        if _STOP_BACKEND is not None:
            _STOP_BACKEND()
            _STOP_BACKEND = None
