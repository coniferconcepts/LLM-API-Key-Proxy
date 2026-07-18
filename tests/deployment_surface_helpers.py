from __future__ import annotations

from pathlib import Path

from proxy_app.runtime_security import (
    RuntimeSecurityConfig,
    build_allowed_hosts,
    build_cors_allowed_origins,
    build_runtime_security_config,
    env_flag_enabled,
    is_loopback_bind_host,
    network_bind_configuration_error,
    runtime_security_source,
    validate_bind_host,
)
from proxy_app.security_middleware import (
    BindApprovalMiddleware,
    TrustedHostMiddleware,
    is_valid_authority_port,
    is_valid_host_name,
    normalize_host_authority,
    send_scope_rejection,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "proxy_app" / "main.py"
BOOTSTRAP_ENV = ROOT / "src" / "proxy_app" / "bootstrap_env.py"
DOCKERFILE = ROOT / "Dockerfile"
README = ROOT / "README.md"
DEPLOYMENT_GUIDE = ROOT / "Deployment guide.md"
COMPOSE_FILES = (
    ROOT / "docker-compose.yml",
    ROOT / "docker-compose.dev.yml",
    ROOT / "docker-compose.tls.yml",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_main_symbols(*names: str) -> dict[str, object]:
    namespace = {
        "_env_flag_enabled": env_flag_enabled,
        "_build_cors_allowed_origins": build_cors_allowed_origins,
        "_build_allowed_hosts": build_allowed_hosts,
        "_build_runtime_security_config": build_runtime_security_config,
        "_is_loopback_bind_host": is_loopback_bind_host,
        "_is_valid_authority_port": is_valid_authority_port,
        "_is_valid_host_name": is_valid_host_name,
        "_network_bind_configuration_error": network_bind_configuration_error,
        "_normalize_host_authority": normalize_host_authority,
        "_runtime_security_source": runtime_security_source,
        "_validate_bind_host": validate_bind_host,
        "_send_scope_rejection": send_scope_rejection,
        "RuntimeSecurityConfig": RuntimeSecurityConfig,
        "BindApprovalMiddleware": BindApprovalMiddleware,
        "TrustedHostMiddleware": TrustedHostMiddleware,
    }
    required_names = {
        "_env_flag_enabled",
        "_build_cors_allowed_origins",
        "_build_allowed_hosts",
        "_build_runtime_security_config",
        "_is_loopback_bind_host",
        "_is_valid_authority_port",
        "_is_valid_host_name",
        "_network_bind_configuration_error",
        "_normalize_host_authority",
        "_runtime_security_source",
        "_send_scope_rejection",
        "RuntimeSecurityConfig",
        *names,
    }
    return {name: namespace[name] for name in required_names}
