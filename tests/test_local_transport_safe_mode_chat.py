from starlette.testclient import TestClient

from test_local_transport_safe_mode import (
    FAKE_XAI_AUTHORITY,
    _block_catalog_fetches,
    _fake_openai_upstream,
    _import_proxy_main,
)


def test_local_transport_safe_mode_routes_exact_xai_chat_to_fake_upstream(
    monkeypatch,
    tmp_path,
):
    with _fake_openai_upstream() as (port, upstream_requests):
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{port}/v1",
        )
        catalog_attempts = _block_catalog_fetches(monkeypatch)

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer proxy-token",
                    "Host": "127.0.0.1",
                    "Origin": "http://127.0.0.1",
                },
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "transport preflight"}],
                    "stream": False,
                },
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fake-ok"
    assert catalog_attempts == []
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["path"] == "/v1/chat/completions"
    assert upstream_requests[0]["authorization"] == "Bearer fake-xai-key"
    assert upstream_requests[0]["body"]["model"] == "grok-4.5"
    assert upstream_requests[0]["body"]["messages"] == [
        {"role": "user", "content": "transport preflight"}
    ]


def test_local_transport_safe_mode_streams_exact_xai_chat_to_fake_upstream(
    monkeypatch,
    tmp_path,
):
    with _fake_openai_upstream() as (port, upstream_requests):
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{port}/v1",
        )
        catalog_attempts = _block_catalog_fetches(monkeypatch)

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "stream preflight"}],
                    "stream": True,
                },
            ) as response:
                response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "fake-ok" in response_body
    assert catalog_attempts == []
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["body"]["stream"] is True


def test_local_transport_safe_mode_rejects_non_xai_chat_before_dispatch(
    monkeypatch,
    tmp_path,
):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args, **_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("safe mode dispatched a non-xAI request")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "antigravity/not-local",
                "messages": [{"role": "user", "content": "must stay local"}],
                "stream": False,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "local_transport_xai_only"
    assert dispatches == 0


def test_local_transport_safe_mode_revalidates_local_base_for_each_chat(
    monkeypatch,
    tmp_path,
):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args, **_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("safe mode dispatched after configuration drift")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
        monkeypatch.setenv("XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "must stay local"}],
                "stream": False,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "local_transport_configuration_changed"
    assert dispatches == 0
