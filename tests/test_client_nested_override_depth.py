import json

import pytest
from starlette.testclient import TestClient

from proxy_app.client_override_guard import find_client_override
from test_client_nested_routing_rejected import _trap_upstream
from test_local_transport_safe_mode import (
    FAKE_XAI_AUTHORITY,
    _block_catalog_fetches,
    _fake_openai_upstream,
    _import_proxy_main,
)

_NESTED_CONTAINERS = (
    "extra_body",
    "metadata",
    "litellm_metadata",
    "litellm_params",
    "litellm_embedding_config",
)
_PROXY_CLASS_FIELDS = (
    "aws_sts_endpoint",
    "aws_web_identity_token",
    "aws_role_name",
    "aws_bedrock_runtime_endpoint",
    "aws_bedrock_project_id",
    "bedrock_tags",
    "vertex_credentials",
    "azure_ad_token",
    "s3_endpoint_url",
    "sagemaker_base_url",
    "deployment_url",
    "nvcf_function_id",
    "use_ssl",
)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"api_base": "http://127.0.0.1:9/v1"}, "api_base"),
        ({"metadata": {"api_base": "http://127.0.0.1:9/v1"}}, "metadata.api_base"),
        (
            {"litellm_metadata": {"azure_ad_token": "stolen"}},
            "litellm_metadata.azure_ad_token",
        ),
        (
            {"litellm_params": {"s3_endpoint_url": "http://127.0.0.1:9"}},
            "litellm_params.s3_endpoint_url",
        ),
        (
            {"litellm_params": {"metadata": {"api_key": "client-key"}}},
            "litellm_params.metadata.api_key",
        ),
        (
            {"litellm_embedding_config": {"api_base": "http://127.0.0.1:9/v1"}},
            "litellm_embedding_config.api_base",
        ),
        (
            {"extra_body": json.dumps({"vertex_credentials": "{}"})},
            "extra_body.vertex_credentials",
        ),
        ({"metadata": {"user_id": "ok"}}, None),
        ({"extra_body": {"wrapper": {"api_base": "http://127.0.0.1:9/v1"}}}, None),
    ],
)
def test_find_client_override_single_level_nested_and_proxy_fields(payload, expected):
    assert find_client_override(payload) == expected


@pytest.mark.parametrize("container", _NESTED_CONTAINERS)
def test_chat_rejects_nested_api_base_without_contacting_trap(monkeypatch, tmp_path, container):
    with _trap_upstream() as (trap_port, trap_requests):
        trap_base = f"http://127.0.0.1:{trap_port}/v1"
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:2465/v1",
        )
        _block_catalog_fetches(monkeypatch)
        dispatches = 0

        async def record_dispatch(*_args, **_kwargs):
            nonlocal dispatches
            dispatches += 1
            raise AssertionError("blocked request dispatched")

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "do not dispatch"}],
                    "stream": False,
                    container: {"api_base": trap_base},
                },
            )

    assert response.status_code == 400
    assert f"{container}.api_base" in response.json()["detail"]
    assert dispatches == 0
    assert trap_requests == []
    assert "fake-xai-key" not in "".join(map(str, trap_requests))


@pytest.mark.parametrize("field", _PROXY_CLASS_FIELDS)
def test_chat_rejects_proxy_class_fields_at_top_level_and_in_metadata(monkeypatch, tmp_path, field):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args, **_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("blocked request dispatched")

    override = True if field == "use_ssl" else "http://127.0.0.1:9"
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
        top_level = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "do not dispatch"}],
                field: override,
            },
        )
        nested = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "do not dispatch"}],
                "metadata": {field: override},
            },
        )

    assert top_level.status_code == 400
    assert field in top_level.json()["detail"]
    assert nested.status_code == 400
    assert f"metadata.{field}" in nested.json()["detail"]
    assert dispatches == 0


def test_chat_rejects_litellm_params_metadata_api_base(monkeypatch, tmp_path):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args, **_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("blocked request dispatched")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "do not dispatch"}],
                "litellm_params": {"metadata": {"api_base": "http://127.0.0.1:9/v1"}},
            },
        )

    assert response.status_code == 400
    assert "litellm_params.metadata.api_base" in response.json()["detail"]
    assert dispatches == 0


def test_chat_allows_benign_metadata_and_does_not_recurse_two_levels(monkeypatch, tmp_path):
    with _fake_openai_upstream() as (port, upstream_requests):
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{port}/v1",
        )
        _block_catalog_fetches(monkeypatch)

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            benign = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "control"}],
                    "stream": False,
                    "metadata": {"user_id": "ok"},
                },
            )
            two_level = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "control"}],
                    "stream": False,
                    "extra_body": {"wrapper": {"api_base": "http://127.0.0.1:9/v1"}},
                },
            )

    assert benign.status_code == 200
    assert two_level.status_code == 200
    assert len(upstream_requests) == 2
    assert all(item["authorization"] == "Bearer fake-xai-key" for item in upstream_requests)
