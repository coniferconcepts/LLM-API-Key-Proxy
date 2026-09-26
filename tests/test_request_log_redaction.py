from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_transport_support = importlib.import_module("test_local_transport_safe_mode")
_block_catalog_fetches = _transport_support._block_catalog_fetches
_import_proxy_main = _transport_support._import_proxy_main

detailed_logger = importlib.import_module("proxy_app.detailed_logger")
transaction_logger = importlib.import_module("rotator_library.transaction_logger")

CANARY = "fake-canary-request-log-7"


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "AcCePt": "application/json",
        "USER-AGENT": "canary-test",
        "X-Request-ID": "test-request-7",
        "AuThOrIzAtIoN": CANARY,
        "X-Auth-Session": CANARY,
        "Cookie": CANARY,
        "X-Credential-ID": CANARY,
        "X-Token-Hint": CANARY,
        "X-Secret-Hint": CANARY,
        "X-Key-Hint": CANARY,
        "X-Unknown": CANARY,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_raw_request_sink_filters_headers_without_mutating_input(monkeypatch, tmp_path, stream):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    headers = _headers()
    original = headers.copy()
    logger = detailed_logger.RawIOLogger()

    logger.log_request(headers, {"stream": stream, "nested": {"extra_headers": _headers()}})
    logger.log_final_response(200, headers, {"model": "test/model"})

    content = (logger.log_dir / "request.json").read_text()
    recorded = json.loads(content)
    response_content = (logger.log_dir / "final_response.json").read_text()
    assert CANARY not in content
    assert CANARY not in response_content
    assert recorded["headers"] == {
        name: headers[name] for name in ("Content-Type", "AcCePt", "USER-AGENT", "X-Request-ID")
    }
    assert recorded["body"]["nested"]["extra_headers"] == recorded["headers"]
    assert headers == original
    assert logger.streaming is stream


def test_transaction_request_sink_filters_nested_headers_and_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(transaction_logger, "_get_transactions_dir", lambda: tmp_path)
    logger = transaction_logger.TransactionLogger("test", "test/model")
    kwargs = {
        "stream": True,
        "extra_headers": _headers(),
        "litellm_params": {"EXTRA_HEADERS": _headers()},
        "steps": [{"request_headers": _headers()}],
        "failure": RuntimeError(f"request data included {CANARY}"),
    }

    logger.log_request(kwargs)
    logger.log_response({"model": "test/model"}, headers=_headers())

    content = (logger.log_dir / "request.json").read_text()
    recorded = json.loads(content)["data"]
    assert CANARY not in content
    assert CANARY not in (logger.log_dir / "response.json").read_text()
    assert recorded["extra_headers"] == recorded["litellm_params"]["EXTRA_HEADERS"]
    assert recorded["steps"][0]["request_headers"] == recorded["extra_headers"]
    assert recorded["failure"] == {"error_type": "RuntimeError"}
    assert kwargs["extra_headers"]["AuThOrIzAtIoN"] == CANARY
    assert logger.streaming is True


@pytest.mark.asyncio
async def test_stream_assembly_exception_does_not_render_request_data(monkeypatch, tmp_path):
    client_module = importlib.import_module("rotator_library.client")
    monkeypatch.setattr(transaction_logger, "_get_transactions_dir", lambda: tmp_path)
    logger = transaction_logger.TransactionLogger("test", "test/model")
    warnings = []
    monkeypatch.setattr(
        client_module.lib_logger, "warning", lambda message: warnings.append(message)
    )

    def failing_assembly(_chunks, _request_data):
        raise RuntimeError(f"request data included {CANARY}")

    monkeypatch.setattr(
        transaction_logger.TransactionLogger,
        "assemble_streaming_response",
        staticmethod(failing_assembly),
    )

    async def chunks():
        yield 'data: {"choices": []}\n\n'

    received = [
        chunk
        async for chunk in client_module.RotatingClient._transaction_logging_stream_wrapper(
            None, chunks(), logger, {"extra_headers": _headers()}
        )
    ]

    assert received == ['data: {"choices": []}\n\n']
    assert warnings == ["TransactionLogger: Failed to assemble/log final response"]
    assert CANARY not in "".join(warnings)


@pytest.mark.parametrize("outcome", ["success", "stream", "rejection", "exception"])
def test_anthropic_request_paths_never_log_header_canary(monkeypatch, tmp_path, caplog, outcome):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1", safe_mode=False)
    _block_catalog_fetches(monkeypatch)
    logs_dir = tmp_path / "request-logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    monkeypatch.setattr(module, "ENABLE_RAW_LOGGING", True)
    console_headers = []
    monkeypatch.setattr(
        module,
        "log_request_to_console",
        lambda **kwargs: console_headers.append(kwargs["headers"]),
    )
    request_body = {
        "model": "xai_oauth/grok-4.5",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 16,
    }
    if outcome == "rejection":
        request_body["metadata"] = {"api_key": "blocked-placeholder"}
    if outcome == "stream":
        request_body["stream"] = True

    async def chunks():
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def completion(*_args, **_kwargs):
        if outcome == "exception":
            raise RuntimeError(f"request headers contained {CANARY}")
        if outcome == "stream":
            return chunks()
        return {"type": "message", "content": [], "model": request_body["model"]}

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "anthropic_messages", completion)
        request_headers = _headers()
        request_headers.pop("AuThOrIzAtIoN")
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "proxy-token", **request_headers},
            json=request_body,
        )

    assert (
        response.status_code
        == {"success": 200, "stream": 200, "rejection": 400, "exception": 500}[outcome]
    )
    artifacts = list((logs_dir / "raw_io").rglob("request.json"))
    assert len(artifacts) == 1
    logged = artifacts[0].read_text()
    assert CANARY not in logged
    if outcome != "rejection":
        assert len(console_headers) == 1
        assert CANARY not in json.dumps(console_headers)
    assert CANARY not in caplog.text
    assert CANARY not in response.text
    assert {name.casefold(): value for name, value in json.loads(logged)["headers"].items()}[
        "x-request-id"
    ] == "test-request-7"
