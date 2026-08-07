from __future__ import annotations

import atexit
import pickle
from typing import Any, Callable, Generic, Literal, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")
Serde = Literal["msgpack", "pickle"]

_shared_context: zmq.Context | None = None
_shared_async_context: zmq.asyncio.Context | None = None


def get_shared_context() -> zmq.Context:
    global _shared_context
    if _shared_context is None:
        _shared_context = zmq.Context(io_threads=1)
    return _shared_context


def get_shared_async_context() -> zmq.asyncio.Context:
    global _shared_async_context
    if _shared_async_context is None:
        _shared_async_context = zmq.asyncio.Context(io_threads=1)
    return _shared_async_context


def shutdown_shared_zmq_contexts() -> None:
    global _shared_context, _shared_async_context
    if _shared_context is not None:
        _shared_context.destroy(linger=0)
        _shared_context = None
    if _shared_async_context is not None:
        _shared_async_context.destroy(linger=0)
        _shared_async_context = None


atexit.register(shutdown_shared_zmq_contexts)


def _serialize(obj: Any, encoder: Callable[[Any], Any], serde: Serde) -> bytes:
    payload = encoder(obj)
    if serde == "msgpack":
        return msgpack.packb(payload, use_bin_type=True)
    if serde == "pickle":
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    raise ValueError(f"Unsupported ZMQ serde: {serde}")


def serialize_zmq_payload(obj: Any, encoder: Callable[[Any], Any], serde: Serde) -> bytes:
    return _serialize(obj, encoder, serde)


def _deserialize(event: bytes, decoder: Callable[[Any], T], serde: Serde) -> T:
    if serde == "msgpack":
        payload = msgpack.unpackb(event, raw=False)
    elif serde == "pickle":
        payload = pickle.loads(event)
    else:
        raise ValueError(f"Unsupported ZMQ serde: {serde}")
    return decoder(payload)


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Any],
        *,
        serde: Serde = "msgpack",
    ):
        self.context = get_shared_context()
        self.socket = self.context.socket(zmq.PUSH)
        self.addr = addr
        if create:
            self.socket.bind(addr)
            self.addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self.socket.connect(addr)
        self.encoder = encoder
        self.serde = serde

    def put(self, obj: T):
        event = serialize_zmq_payload(obj, self.encoder, self.serde)
        self.socket.send(event, copy=False)

    def serialize(self, obj: T) -> bytes:
        return serialize_zmq_payload(obj, self.encoder, self.serde)

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def stop(self):
        self.socket.close()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Any],
        *,
        serde: Serde = "msgpack",
    ):
        self.context = get_shared_async_context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder
        self.serde = serde

    async def put(self, obj: T):
        event = serialize_zmq_payload(obj, self.encoder, self.serde)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Any], T],
        *,
        serde: Serde = "msgpack",
    ):
        self.context = get_shared_context()
        self.socket = self.context.socket(zmq.PULL)
        self.addr = addr
        if create:
            self.socket.bind(addr)
            self.addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self.socket.connect(addr)
        self.decoder = decoder
        self.serde = serde

    def get(self, timeout_ms: int | None = None) -> T:
        if timeout_ms is not None:
            if self.socket.poll(timeout=timeout_ms) == 0:
                raise TimeoutError(f"ZmqPullQueue.get timed out after {timeout_ms}ms")
        event = self.socket.recv()
        return _deserialize(event, self.decoder, self.serde)

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return _deserialize(raw, self.decoder, self.serde)

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Any], T],
        *,
        serde: Serde = "msgpack",
    ):
        self.context = get_shared_async_context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder
        self.serde = serde

    async def get(self) -> T:
        event = await self.socket.recv()
        return _deserialize(event, self.decoder, self.serde)

    def stop(self):
        self.socket.close()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Any],
        *,
        serde: Serde = "msgpack",
    ):
        self.context = get_shared_context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder
        self.serde = serde

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = serialize_zmq_payload(obj, self.encoder, self.serde)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Any], T],
        *,
        serde: Serde = "msgpack",
    ):
        self.context = get_shared_context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder
        self.serde = serde

    def get(self) -> T:
        event = self.socket.recv()
        return _deserialize(event, self.decoder, self.serde)

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
