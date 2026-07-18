"""OAuth credential initialization isolated from the ASGI lifespan boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import json
import logging
import time
from typing import Any, Protocol


class OAuthProvider(Protocol):
    async def initialize_token(self, path: str) -> None: ...


ProviderFactory = Callable[[], OAuthProvider]
SafeExceptionLogger = Callable[[str, BaseException, int], None]
CredentialResult = tuple[str, str, str | None, BaseException | None]


async def initialize_oauth_credentials(
    oauth_credentials: Mapping[str, list[str]],
    provider_plugins: Mapping[str, ProviderFactory],
    log_safe_exception: SafeExceptionLogger,
) -> dict[str, list[str]]:
    processed_emails: dict[str, dict[str, str]] = {}
    credentials_to_initialize: dict[str, list[str]] = {}

    for provider, paths in oauth_credentials.items():
        provider_paths = credentials_to_initialize.setdefault(provider, [])
        for path in paths:
            if path.startswith("env://"):
                provider_paths.append(path)
                continue
            try:
                with open(path, encoding="utf-8") as credential_file:
                    data: dict[str, Any] = json.load(credential_file)
                metadata = data.get("_proxy_metadata", {})
                email = metadata.get("email")
                if email:
                    provider_paths_by_email = processed_emails.setdefault(email, {})
                    if provider in provider_paths_by_email:
                        logging.warning("Duplicate OAuth credential skipped during pre-scan")
                        continue
                    provider_paths_by_email[provider] = path
                provider_paths.append(path)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                log_safe_exception("OAuth metadata pre-read", exc, 500)
                provider_paths.append(path)

    async def process_credential(
        provider: str,
        path: str,
        provider_instance: OAuthProvider,
    ) -> CredentialResult:
        try:
            await provider_instance.initialize_token(path)
            get_user_info = getattr(provider_instance, "get_user_info", None)
            if get_user_info is None:
                return provider, path, None, None
            user_info = await get_user_info(path)
            return provider, path, user_info.get("email"), None
        except Exception as exc:  # noqa: BLE001 - provider plugins are an external boundary
            log_safe_exception("OAuth token initialization", exc, 500)
            return provider, path, None, exc

    tasks: list[Awaitable[CredentialResult]] = []
    for provider, paths in credentials_to_initialize.items():
        if not paths:
            continue
        provider_plugin_factory = provider_plugins.get(provider)
        if provider_plugin_factory is None:
            continue
        provider_instance = provider_plugin_factory()
        tasks.extend(process_credential(provider, path, provider_instance) for path in paths)

    gathered_results = await asyncio.gather(*tasks, return_exceptions=True)
    final_oauth_credentials: dict[str, list[str]] = {}
    for result in gathered_results:
        if isinstance(result, BaseException):
            log_safe_exception("OAuth credential task", result, 500)
            continue
        provider, path, email, error = result
        if error:
            continue
        if email is None:
            final_oauth_credentials.setdefault(provider, []).append(path)
            continue
        if not email:
            logging.warning("OAuth credential email unavailable; treating credential as unique")
            final_oauth_credentials.setdefault(provider, []).append(path)
            continue

        provider_paths_by_email = processed_emails.setdefault(email, {})
        if provider in provider_paths_by_email and provider_paths_by_email[provider] != path:
            logging.warning("Duplicate OAuth credential skipped after initialization")
            continue
        provider_paths_by_email[provider] = path
        final_oauth_credentials.setdefault(provider, []).append(path)
        if not path.startswith("env://"):
            try:
                with open(path, "r+", encoding="utf-8") as credential_file:
                    updated_data: dict[str, Any] = json.load(credential_file)
                    metadata = updated_data.get("_proxy_metadata", {})
                    metadata["email"] = email
                    metadata["last_check_timestamp"] = time.time()
                    updated_data["_proxy_metadata"] = metadata
                    credential_file.seek(0)
                    json.dump(updated_data, credential_file, indent=2)
                    credential_file.truncate()
            except Exception as exc:  # noqa: BLE001 - credential files are an external boundary
                log_safe_exception("OAuth metadata update", exc, 500)

    logging.info("OAuth credential processing complete.")
    return final_oauth_credentials
