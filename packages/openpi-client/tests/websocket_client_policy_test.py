from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class FakeConnection:
    def recv(self):
        return msgpack_numpy.packb({"protocol": "sonic_vla_v1"})


def test_policy_websocket_bypasses_process_proxy(monkeypatch):
    def connect(_uri, *, compression, max_size, additional_headers, proxy="process-proxy"):
        assert compression is None
        assert max_size is None
        assert additional_headers is None
        assert proxy is None
        return FakeConnection()

    monkeypatch.setattr("websockets.sync.client.connect", connect)
    client = WebsocketClientPolicy(host="192.168.123.10", port=8000)

    assert client.get_server_metadata()["protocol"] == "sonic_vla_v1"


def test_policy_websocket_supports_older_websockets_without_proxy_keyword(monkeypatch):
    def connect(_uri, *, compression, max_size, additional_headers):
        return FakeConnection()

    monkeypatch.setattr("websockets.sync.client.connect", connect)
    client = WebsocketClientPolicy(host="127.0.0.1", port=8000)

    assert client.get_server_metadata()["protocol"] == "sonic_vla_v1"
