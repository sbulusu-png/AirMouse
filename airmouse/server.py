"""Local WebSocket bridge between the engine and the UI.

Security: binds to 127.0.0.1 only and requires a per-run random token in the URL
(ws://127.0.0.1:PORT/?token=...), so web pages open in a browser can't connect and
watch the camera or change settings.

Plain HTTP on the same port:
  GET  /, /<file>   the UI files from frontend/ (no secrets in them; the window is opened at
                    http://127.0.0.1:PORT/?token=... and the page connects back with it)
  GET  /show        with header "X-AirMouse: show": brings the window up. Used by a second
                    launch of the app. Web pages can't send that header to us (CORS).

Protocol
  server -> client
    text   {"type": "hello", "config", "schema", "connections", "hotkeys"}   on connect
    text   {"type": "state", ...}                     every frame (latest-only; slow clients skip)
    binary JPEG of the mirrored camera frame          right after each state message
    text   {"type": "config", "config": {...}}        after any settings change
    text   {"type": "error", "message": "..."}        reply to an invalid command
  client -> server
    {"type": "set_config", "values": {"name": number, ...}}
    {"type": "reset_config"}
    {"type": "set_enabled", "enabled": true|false}
    {"type": "set_autostart", "enabled": true|false}   start with Windows; answered with
        {"type": "autostart", "enabled": ...} to every client
    {"type": "calibrate"} / {"type": "calibrate_cancel"}   fit the control area to the
        user's reach; progress and the proposed settings come back in state.calibration
"""

import asyncio
import json
import mimetypes
import secrets
import threading
from collections.abc import Callable
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from . import autostart
from .config import PROJECT_ROOT
from .engine import Engine
from .hotkeys import QUIT_LABEL, TOGGLE_LABEL
from .instance import DEFAULT_PORT, SHOW_HEADER
from .settings import save_config, schema
from .tracker import HAND_CONNECTIONS

STATIC_DIR = (PROJECT_ROOT / "frontend").resolve()
STATIC_TYPES = {".html", ".css", ".js", ".json", ".svg", ".png", ".ico", ".woff2"}
EXTRA_TYPES = {".woff2": "font/woff2", ".js": "text/javascript"}  # not in every mimetypes db


class Server:
    def __init__(self, engine: Engine, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.token = secrets.token_urlsafe(16)
        self._clients: set[ServerConnection] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Future[None] | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._latest: tuple[str, bytes | None] | None = None
        self._seq = 0
        self._new_frame: asyncio.Event | None = None
        self.streaming = True                     # False while the UI window is hidden
        self.on_show: Callable[[], None] | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/?token={self.token}"

    @property
    def http_url(self) -> str:
        """The UI page, carrying the token it needs to connect back."""
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def set_streaming(self, on: bool) -> None:
        """Stop sending state and video while nobody can see them (window hidden)."""
        self.streaming = on
        self.engine.want_jpeg = on and bool(self._clients)

    def start(self) -> None:
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()),
                                        name="airmouse-server", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._startup_error is not None:
            raise RuntimeError(f"WebSocket server failed to start: {self._startup_error}")

    def stop(self) -> None:
        loop, stopped = self._loop, self._stopped
        if loop is not None and stopped is not None and not loop.is_closed():
            loop.call_soon_threadsafe(lambda: stopped.done() or stopped.set_result(None))
        if self._thread is not None:
            self._thread.join(timeout=5)

    def publish(self, state: dict[str, Any], jpeg: bytes | None) -> None:
        """Called from the engine thread once per frame."""
        loop = self._loop
        if loop is not None and self._clients and self.streaming:
            loop.call_soon_threadsafe(self._set_latest, json.dumps(state), jpeg)

    # ---- event loop thread ---------------------------------------------------------------

    def _set_latest(self, text: str, jpeg: bytes | None) -> None:
        self._latest = (text, jpeg)
        self._seq += 1
        if self._new_frame is not None:
            self._new_frame.set()               # wake every waiting sender...
        self._new_frame = asyncio.Event()       # ...and give the next frame a fresh event

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = self._loop.create_future()
        self._new_frame = asyncio.Event()
        try:
            server = await serve(self._handler, self.host, self.port,
                                 process_request=self._check_token,
                                 compression=None,          # JPEG is already compressed
                                 max_size=64 * 1024)        # client messages are tiny
        except OSError as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self.port = server.sockets[0].getsockname()[1]  # resolves port=0 to the real port
        self._ready.set()
        async with server:
            await self._stopped

    def _check_token(self, conn: ServerConnection, request: Request) -> Response | None:
        if request.headers.get("Upgrade", "").lower() != "websocket":
            return self._http(request)
        token = parse_qs(urlsplit(request.path).query).get("token", [""])[0]
        if not secrets.compare_digest(token, self.token):
            return _response(HTTPStatus.FORBIDDEN, b"Forbidden\n")
        return None

    def _http(self, request: Request) -> Response:
        path = unquote(urlsplit(request.path).path)
        if path == "/show":
            # websockets only routes GET; a second app launch sends this with our header.
            if request.headers.get(SHOW_HEADER) != "show":
                return _response(HTTPStatus.FORBIDDEN, b"Forbidden\n")
            if self.on_show is not None:
                self.on_show()
            return _response(HTTPStatus.OK, b"shown\n")
        file = (STATIC_DIR / (path.lstrip("/") or "index.html")).resolve()
        if (STATIC_DIR not in file.parents or file.suffix not in STATIC_TYPES
                or not file.is_file()):
            return _response(HTTPStatus.NOT_FOUND, b"Not found\n")
        kind = (EXTRA_TYPES.get(file.suffix) or mimetypes.guess_type(file.name)[0]
                or "application/octet-stream")
        if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
            kind += "; charset=utf-8"
        return _response(HTTPStatus.OK, file.read_bytes(), kind, cache=False)

    async def _handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        self.engine.want_jpeg = self.streaming
        sender = asyncio.create_task(self._send_frames(ws))
        try:
            await ws.send(json.dumps({
                "type": "hello",
                "config": self.engine.config_dict(),
                "schema": schema(),
                "connections": [list(c) for c in HAND_CONNECTIONS],
                "hotkeys": {"toggle": TOGGLE_LABEL, "quit": QUIT_LABEL},
                "autostart": autostart.is_enabled(),
            }))
            async for message in ws:
                await self._on_message(ws, message)
        except ConnectionClosed:
            pass
        finally:
            sender.cancel()
            self._clients.discard(ws)
            self.engine.want_jpeg = bool(self._clients)

    async def _send_frames(self, ws: ServerConnection) -> None:
        seen = -1
        try:
            while True:
                event = self._new_frame
                if self._seq == seen or self._latest is None:
                    if event is not None:
                        await event.wait()
                    continue
                seen = self._seq
                text, jpeg = self._latest
                await ws.send(text)          # awaiting the send drops frames for slow clients
                if jpeg is not None:
                    await ws.send(jpeg)
        except ConnectionClosed:
            pass

    async def _on_message(self, ws: ServerConnection, message: str | bytes) -> None:
        try:
            msg = json.loads(message)
            if not isinstance(msg, dict):
                raise ValueError("message must be a JSON object")
            kind = msg.get("type")
            if kind == "set_config":
                values = msg.get("values")
                if not isinstance(values, dict):
                    raise ValueError("set_config needs a 'values' object")
                self._config_changed(self.engine.update_config(values))
            elif kind == "reset_config":
                self._config_changed(self.engine.reset_config())
            elif kind == "set_enabled":
                self.engine.set_enabled(bool(msg.get("enabled")))
            elif kind == "set_autostart":
                autostart.set_enabled(bool(msg.get("enabled")))
                broadcast(self._clients, json.dumps({"type": "autostart",
                                                     "enabled": autostart.is_enabled()}))
            elif kind == "calibrate":
                self.engine.start_calibration()
            elif kind == "calibrate_cancel":
                self.engine.cancel_calibration()
            else:
                raise ValueError(f"unknown message type: {kind!r}")
        except ValueError as exc:  # includes json.JSONDecodeError
            await ws.send(json.dumps({"type": "error", "message": str(exc)}))
        except OSError as exc:     # settings applied, but saving them failed
            await ws.send(json.dumps({"type": "error",
                                      "message": f"Setting applied but not saved: {exc}"}))

    def _config_changed(self, changed: dict[str, Any]) -> None:
        # Broadcast first, so every UI shows the live values even if saving fails below.
        broadcast(self._clients, json.dumps({"type": "config",
                                             "config": self.engine.config_dict()}))
        if changed:
            save_config(self.engine.cfg)


def _response(status: HTTPStatus, body: bytes, content_type: str = "text/plain; charset=utf-8",
              cache: bool = True) -> Response:
    headers = Headers({"Content-Type": content_type, "Content-Length": str(len(body)),
                       "Connection": "close"})
    if not cache:
        headers["Cache-Control"] = "no-store"
    return Response(status.value, status.phrase, headers, body)
