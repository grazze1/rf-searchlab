import socket

import pytest


@pytest.fixture(autouse=True)
def offline_and_local(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_SEARCHLAB_HOME", str(tmp_path))
    (tmp_path / "workspace.json").write_text("{}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in tests")

    original_connect = socket.socket.connect

    def local_event_loop_only(sock, address):
        # Windows asyncio implements its self-pipe with a local ephemeral socket pair.
        if isinstance(address, tuple) and address[0] == "127.0.0.1" and address[1] >= 49152:
            return original_connect(sock, address)
        return forbidden()

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", local_event_loop_only)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
