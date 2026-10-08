"""WebSocket protocol tests. The engine is constructed but never started: no camera needed."""

import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

import airmouse.server as server_mod
from airmouse.config import Config
from airmouse.engine import Engine
from airmouse.server import Server


@pytest.fixture
def server(tmp_path, monkeypatch):
    real_save = server_mod.save_config
    monkeypatch.setattr(server_mod, "save_config", lambda cfg: real_save(cfg, tmp_path / "s.json"))
    engine = Engine(Config(), enabled=False)
    srv = Server(engine, port=0)
    srv.start()
    yield srv
    srv.stop()


async def recv_type(ws, kind: str, limit: int = 50) -> dict:
    for _ in range(limit):
        msg = await asyncio.wait_for(ws.recv(), 5)
        if isinstance(msg, str) and json.loads(msg)["type"] == kind:
            return json.loads(msg)
    raise AssertionError(f"no {kind!r} message")


def test_wrong_or_missing_token_is_forbidden(server):
    async def attempt(url):
        with pytest.raises(InvalidStatus) as exc:
            async with connect(url):
                pass
        assert exc.value.response.status_code == 403

    base = f"ws://127.0.0.1:{server.port}/"
    asyncio.run(attempt(base))
    asyncio.run(attempt(base + "?token=wrong"))


def test_hello_describes_config_schema_and_skeleton(server):
    async def go():
        async with connect(server.url) as ws:
            return json.loads(await ws.recv())

    hello = asyncio.run(go())
    assert hello["type"] == "hello"
    assert hello["config"]["push_threshold"] == Config().push_threshold
    assert {p["name"] for p in hello["schema"]} == set(hello["config"])
    assert len(hello["connections"]) == 21
    assert server.engine.want_jpeg is False  # reset once the client left


def test_published_state_and_frame_reach_client(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.recv()  # hello
            assert server.engine.want_jpeg is True
            for i in range(3):
                server.publish({"type": "state", "n": i}, b"\xff\xd8jpeg")
            state = await recv_type(ws, "state")
            frame = await asyncio.wait_for(ws.recv(), 5)
            return state, frame

    state, frame = asyncio.run(go())
    assert state["type"] == "state" and frame == b"\xff\xd8jpeg"


def test_set_config_clamps_persists_and_broadcasts(server, tmp_path):
    async def go():
        async with connect(server.url) as a, connect(server.url) as b:
            await a.recv()
            await b.recv()
            await a.send(json.dumps({"type": "set_config",
                                     "values": {"push_threshold": 0.35, "cursor_speed": 99}}))
            return await recv_type(a, "config"), await recv_type(b, "config")

    ca, cb = asyncio.run(go())
    assert ca == cb
    assert ca["config"]["push_threshold"] == 0.35 and ca["config"]["cursor_speed"] == 5.0
    assert json.loads((tmp_path / "s.json").read_text())["push_threshold"] == 0.35


@pytest.mark.parametrize("message", [
    "not json",
    "[]",
    json.dumps({"type": "nope"}),
    json.dumps({"type": "set_config", "values": {"bogus": 1}}),
    json.dumps({"type": "set_config", "values": "x"}),
])
def test_invalid_messages_get_error_and_change_nothing(server, message):
    async def go():
        async with connect(server.url) as ws:
            await ws.recv()
            await ws.send(message)
            return await recv_type(ws, "error")

    assert asyncio.run(go())["message"]
    assert server.engine.cfg.push_threshold == Config().push_threshold


def test_nan_from_client_is_rejected(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.recv()
            await ws.send('{"type": "set_config", "values": {"smoothing_alpha": NaN}}')
            return await recv_type(ws, "error")

    assert "finite" in asyncio.run(go())["message"]
    assert server.engine.cfg.smoothing_alpha == Config().smoothing_alpha


def test_save_failure_reports_error_and_keeps_connection(server, monkeypatch):
    def locked(cfg):
        raise PermissionError("settings.json is locked")

    monkeypatch.setattr(server_mod, "save_config", locked)

    async def go():
        async with connect(server.url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "set_config", "values": {"smoothing_alpha": 0.5}}))
            config = await recv_type(ws, "config")
            error = await recv_type(ws, "error")
            await ws.send(json.dumps({"type": "set_enabled", "enabled": True}))  # still open
            return config, error

    config, error = asyncio.run(go())
    assert config["config"]["smoothing_alpha"] == 0.5
    assert "not saved" in error["message"]


def test_set_enabled_and_reset_config(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "set_enabled", "enabled": True}))
            await ws.send(json.dumps({"type": "set_config", "values": {"smoothing_alpha": 0.9}}))
            await recv_type(ws, "config")
            await ws.send(json.dumps({"type": "reset_config"}))
            return await recv_type(ws, "config")

    reset = asyncio.run(go())
    assert server.engine.enabled is True
    assert reset["config"]["smoothing_alpha"] == Config().smoothing_alpha


# --- Plain HTTP: UI files and the single-instance "show" request ---------------------------

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def http_get(server, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, "", b""


def test_serves_ui_files(server):
    status, kind, body = http_get(server, "/?token=whatever")
    assert status == 200 and kind.startswith("text/html") and b"AirMouse" in body
    status, kind, _ = http_get(server, "/app.js")
    assert status == 200 and "javascript" in kind
    status, kind, body = http_get(server, "/fonts/SchibstedGrotesk-latin.woff2")
    assert status == 200 and kind == "font/woff2" and body[:4] == b"wOF2"


@pytest.mark.parametrize("path", ["/../airmouse/config.py", "/%2e%2e/airmouse/server.py",
                                  "/nope.html", "/demo-hand.py"])
def test_never_serves_files_outside_the_ui(server, path):
    assert http_get(server, path)[0] == 404


def test_show_requires_the_app_header(server):
    calls = []
    server.on_show = lambda: calls.append(1)
    assert http_get(server, "/show")[0] == 403                      # e.g. a web page
    assert calls == []
    assert http_get(server, "/show", {"X-AirMouse": "show"})[0] == 200
    assert calls == [1]


def test_hidden_window_pauses_streaming(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.recv()                                         # hello
            server.set_streaming(False)
            assert server.engine.want_jpeg is False                 # no JPEG encoding either
            server.publish({"type": "state", "n": 1}, b"jpeg")
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), 0.3)              # nothing arrives
            server.set_streaming(True)
            server.publish({"type": "state", "n": 2}, b"jpeg")
            return json.loads(await asyncio.wait_for(ws.recv(), 5))

    assert asyncio.run(go())["n"] == 2                              # resumes when shown


# --- Start with Windows ---------------------------------------------------------------------------

import winreg  # noqa: E402

import airmouse.autostart as autostart_mod  # noqa: E402

TEST_KEY = r"Software\AirMouseTest\Run"   # never the real startup list


@pytest.fixture
def test_run_key(monkeypatch):
    monkeypatch.setattr(autostart_mod, "RUN_KEY", TEST_KEY)
    yield
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, TEST_KEY)
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\AirMouseTest")
    except OSError:
        pass


def test_autostart_switch_writes_and_removes_the_entry(test_run_key):
    assert autostart_mod.is_enabled() is False
    autostart_mod.set_enabled(True)
    assert autostart_mod.is_enabled() is True
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, TEST_KEY) as key:
        value, _ = winreg.QueryValueEx(key, "AirMouse")
    assert "pythonw.exe" in value and "AirMouse.pyw" in value and "--minimized" in value
    autostart_mod.set_enabled(False)
    assert autostart_mod.is_enabled() is False
    autostart_mod.set_enabled(False)                          # turning off twice is fine


def test_autostart_over_the_protocol(server, test_run_key):
    async def go():
        async with connect(server.url) as a, connect(server.url) as b:
            hello = json.loads(await a.recv())
            await b.recv()
            await a.send(json.dumps({"type": "set_autostart", "enabled": True}))
            return hello["autostart"], await recv_type(a, "autostart"), await recv_type(b, "autostart")

    before, mine, other = asyncio.run(go())
    assert before is False and mine["enabled"] is True and other == mine


def test_calibrate_messages_reach_the_engine(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "calibrate"}))
            await asyncio.sleep(0.2)
            started = server.engine._calibration is not None
            await ws.send(json.dumps({"type": "calibrate_cancel"}))
            await asyncio.sleep(0.2)
            return started, server.engine._calibration is None

    assert asyncio.run(go()) == (True, True)
