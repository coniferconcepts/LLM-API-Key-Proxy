# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/openai_chatgpt_auth_base.py

import asyncio
import base64
import json
import logging
import os
import re
import time
import webbrowser
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from ..error_handler import CredentialNeedsReauthError
from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")

console = Console()

ENV_PREFIX = "OPENAI_CHATGPT"
REFRESH_EXPIRY_BUFFER_SECONDS = 10 * 60
CHATGPT_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"


@dataclass
class OpenAIChatGPTCredentialSetupResult:
    """
    Standardized result structure for ChatGPT credential setup operations.
    """

    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    account_id: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


class OpenAIChatGPTAuthBase:
    """
    Base class for ChatGPT browser-session OAuth-like authentication.

    This provider uses a manual token entry flow for browser-extracted tokens,
    then manages refresh and rotation using the same queue/lock pattern used by
    other OAuth providers.
    """

    ENV_PREFIX = ENV_PREFIX

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}

        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None

        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        self._queued_credentials: set = set()
        self._unavailable_credentials: Dict[str, float] = {}
        self._unavailable_ttl_seconds: int = 360
        self._queue_tracking_lock = asyncio.Lock()
        self._queue_retry_count: Dict[str, int] = {}

        self._refresh_timeout_seconds: int = 20
        self._refresh_interval_seconds: int = 15
        self._refresh_max_retries: int = 3
        self._reauth_timeout_seconds: int = 300

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        if not path.startswith("env://"):
            return None

        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load ChatGPT credentials from environment variables.

        Supports:
        - Legacy: OPENAI_CHATGPT_ACCESS_TOKEN
        - Numbered: OPENAI_CHATGPT_1_ACCESS_TOKEN, OPENAI_CHATGPT_2_ACCESS_TOKEN, ...
        """
        if credential_index and credential_index != "0":
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = self.ENV_PREFIX
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        if not access_token:
            return None

        expiry_str = os.getenv(
            f"{prefix}_EXPIRES_AT", os.getenv(f"{prefix}_EXPIRY_DATE", "0")
        )
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            lib_logger.warning(
                f"Invalid {prefix}_EXPIRES_AT/{prefix}_EXPIRY_DATE value: {expiry_str}, using 0"
            )
            expiry_date = 0

        token_payload = self._decode_jwt_payload(access_token)
        account_id = os.getenv(f"{prefix}_ACCOUNT_ID") or self._extract_account_id(
            access_token, token_payload
        )
        email = os.getenv(f"{prefix}_EMAIL") or self._extract_email(token_payload)

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "account_id": account_id,
            "_proxy_metadata": {
                "email": email or default_email,
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"ChatGPT credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load ChatGPT credentials from '{path}': {e}")

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise IOError(
                    f"Environment variables for {self.ENV_PREFIX} credential index {credential_index} not found"
                )

            try:
                return await self._read_creds_from_file(path)
            except IOError:
                env_creds = self._load_from_env()
                if env_creds:
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise

    async def _save_credentials(self, path: str, creds: Dict[str, Any]) -> bool:
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            self._credentials_cache[path] = creds
            return True

        if not safe_write_json(
            path,
            creds,
            lib_logger,
            secure_permissions=True,
            buffer_on_failure=False,
        ):
            lib_logger.error(
                f"Failed to persist ChatGPT credentials for '{Path(path).name}'."
            )
            return False

        self._credentials_cache[path] = creds
        return True

    def _decode_jwt_payload(self, token: Optional[str]) -> Dict[str, Any]:
        if not token or token.count(".") < 2:
            return {}

        try:
            payload_part = token.split(".")[1]
            padding = "=" * (-len(payload_part) % 4)
            payload = base64.urlsafe_b64decode(payload_part + padding).decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}

        return {}

    def _extract_account_id(
        self, access_token: Optional[str], payload: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        data = payload or self._decode_jwt_payload(access_token)

        for key in [
            "account_id",
            "chatgpt_account_id",
            "org_id",
            "organization_id",
            "sub",
            "user_id",
        ]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        nested_profile = data.get("https://api.openai.com/profile")
        if isinstance(nested_profile, dict):
            for key in ["account_id", "id", "sub"]:
                value = nested_profile.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _extract_email(self, payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not payload:
            return None

        email = payload.get("email")
        if isinstance(email, str) and email.strip():
            return email.strip()

        nested_profile = payload.get("https://api.openai.com/profile")
        if isinstance(nested_profile, dict):
            email = nested_profile.get("email")
            if isinstance(email, str) and email.strip():
                return email.strip()

        return None

    def _extract_expiry_from_access_token(self, access_token: str) -> Optional[float]:
        payload = self._decode_jwt_payload(access_token)
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
        return None

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry_timestamp = float(creds.get("expiry_date", 0) or 0)
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        expiry_timestamp = float(creds.get("expiry_date", 0) or 0)
        return expiry_timestamp < time.time()

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            creds = await self._load_credentials(path)
            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise CredentialNeedsReauthError(
                    credential_path=path,
                    message=f"Missing refresh token for '{Path(path).name}'.",
                )

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
            }
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": os.getenv(
                    f"{self.ENV_PREFIX}_CLIENT_ID", "chatgpt-web-client"
                ),
            }

            max_retries = 3
            last_error: Optional[Exception] = None
            token_data: Optional[Dict[str, Any]] = None

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            CHATGPT_TOKEN_ENDPOINT,
                            headers=headers,
                            json=payload,
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        data = response.json()
                        if isinstance(data, dict):
                            token_data = data
                            break
                        raise ValueError("Invalid refresh response format")

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code

                        if status_code in (400, 401, 403):
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Credential '{Path(path).name}' requires re-auth (HTTP {status_code}).",
                            )

                        if status_code == 429 and attempt < max_retries - 1:
                            retry_after = int(e.response.headers.get("Retry-After", 30))
                            await asyncio.sleep(retry_after)
                            continue

                        if 500 <= status_code < 600 and attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                            continue

                        raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise

            if token_data is None:
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                backoff_seconds = min(300, 30 * (2 ** self._refresh_failures[path]))
                self._next_refresh_after[path] = time.time() + backoff_seconds
                raise last_error or RuntimeError("Token refresh failed")

            new_access_token = token_data.get("access_token")
            if not new_access_token:
                raise ValueError("Refreshed token payload missing access_token")

            creds["access_token"] = new_access_token
            creds["refresh_token"] = token_data.get("refresh_token", refresh_token)

            expires_at = token_data.get("expires_at")
            expires_in = token_data.get("expires_in")
            if isinstance(expires_at, (int, float)):
                creds["expiry_date"] = float(expires_at)
            elif isinstance(expires_in, (int, float)):
                creds["expiry_date"] = time.time() + float(expires_in)
            else:
                token_exp = self._extract_expiry_from_access_token(new_access_token)
                creds["expiry_date"] = token_exp or (time.time() + 30 * 60)

            token_payload = self._decode_jwt_payload(new_access_token)
            creds["account_id"] = creds.get("account_id") or self._extract_account_id(
                new_access_token, token_payload
            )

            metadata = creds.setdefault("_proxy_metadata", {})
            metadata["email"] = metadata.get("email") or self._extract_email(
                token_payload
            )
            metadata["last_check_timestamp"] = time.time()

            self._refresh_failures.pop(path, None)
            self._next_refresh_after.pop(path, None)

            if not await self._save_credentials(path, creds):
                raise IOError(f"Failed to persist refreshed credentials for '{path}'.")

            return self._credentials_cache[path]

    async def _get_lock(self, path: str) -> asyncio.Lock:
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def is_credential_available(self, path: str) -> bool:
        if path in self._unavailable_credentials:
            marked_time = self._unavailable_credentials.get(path)
            if marked_time is not None:
                now = time.time()
                if now - marked_time > self._unavailable_ttl_seconds:
                    self._unavailable_credentials.pop(path, None)
                    self._queued_credentials.discard(path)
                else:
                    return False

        creds = self._credentials_cache.get(path)
        if creds and self._is_token_truly_expired(creds):
            if path not in self._queued_credentials:
                asyncio.create_task(
                    self._queue_refresh(path, force=True, needs_reauth=False)
                )
            return False

        return True

    async def _ensure_queue_processor_running(self):
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

    async def _ensure_reauth_processor_running(self):
        if self._reauth_processor_task is None or self._reauth_processor_task.done():
            self._reauth_processor_task = asyncio.create_task(
                self._process_reauth_queue()
            )

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        if not needs_reauth:
            now = time.time()
            if (
                path in self._next_refresh_after
                and now < self._next_refresh_after[path]
            ):
                return

        async with self._queue_tracking_lock:
            if path in self._queued_credentials:
                return

            self._queued_credentials.add(path)

            if needs_reauth:
                self._unavailable_credentials[path] = time.time()
                await self._reauth_queue.put(path)
                await self._ensure_reauth_processor_running()
            else:
                await self._refresh_queue.put((path, force))
                await self._ensure_queue_processor_running()

    async def _process_refresh_queue(self):
        while True:
            path = None
            try:
                try:
                    path, force = await asyncio.wait_for(
                        self._refresh_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    async with self._queue_tracking_lock:
                        self._queue_retry_count.clear()
                    self._queue_processor_task = None
                    return

                try:
                    creds = self._credentials_cache.get(path)
                    if creds and not self._is_token_expired(creds):
                        self._queue_retry_count.pop(path, None)
                        continue

                    try:
                        async with asyncio.timeout(self._refresh_timeout_seconds):
                            await self._refresh_token(path, force=force)
                        self._queue_retry_count.pop(path, None)

                    except asyncio.TimeoutError:
                        await self._handle_refresh_failure(path, force, "timeout")

                    except CredentialNeedsReauthError:
                        self._queue_retry_count.pop(path, None)
                        async with self._queue_tracking_lock:
                            self._queued_credentials.discard(path)
                        await self._queue_refresh(path, force=True, needs_reauth=True)

                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        if status_code in (400, 401, 403):
                            self._queue_retry_count.pop(path, None)
                            async with self._queue_tracking_lock:
                                self._queued_credentials.discard(path)
                            await self._queue_refresh(
                                path, force=True, needs_reauth=True
                            )
                        else:
                            await self._handle_refresh_failure(
                                path, force, f"HTTP {status_code}"
                            )

                    except Exception as e:
                        await self._handle_refresh_failure(path, force, str(e))

                finally:
                    async with self._queue_tracking_lock:
                        if (
                            path in self._queued_credentials
                            and self._queue_retry_count.get(path, 0) == 0
                        ):
                            self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()

                await asyncio.sleep(self._refresh_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                lib_logger.error(f"Error in ChatGPT refresh queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            lib_logger.error(
                f"Max retries reached for '{Path(path).name}' (last error: {error})."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        await self._refresh_queue.put((path, force))

    async def _process_reauth_queue(self):
        while True:
            path = None
            try:
                try:
                    path = await asyncio.wait_for(
                        self._reauth_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    self._reauth_processor_task = None
                    return

                try:
                    await self.initialize_token(path, force_interactive=True)
                except Exception as e:
                    lib_logger.error(f"ChatGPT re-auth failed for '{path}': {e}")
                finally:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                    self._reauth_queue.task_done()

            except asyncio.CancelledError:
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                break
            except Exception as e:
                lib_logger.error(f"Error in ChatGPT re-auth queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)

    async def _perform_interactive_oauth(
        self, path: Optional[str], creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Manual token-entry flow for ChatGPT browser session tokens.
        """
        is_headless = is_headless_environment()

        panel_text = Text.from_markup(
            "1. Open ChatGPT in your browser and sign in with your Plus/team account.\n"
            "2. Extract session OAuth tokens (access_token + refresh_token).\n"
            "3. Paste either a JSON object with access_token/refresh_token/expires_at/email,\n"
            "   or paste raw access_token JWT and then enter the remaining values."
        )
        console.print(
            Panel(
                panel_text,
                title=f"ChatGPT OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                style="bold blue",
            )
        )

        if not is_headless:
            try:
                webbrowser.open("https://chatgpt.com/")
            except Exception:
                pass

        raw_entry = Prompt.ask(
            "Paste token JSON or access token JWT",
        ).strip()

        parsed: Dict[str, Any] = {}
        access_token: Optional[str] = None
        refresh_token: Optional[str] = None
        expiry_date: Optional[float] = None
        email: Optional[str] = None

        if raw_entry.startswith("{"):
            try:
                parsed = json.loads(raw_entry)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON token payload: {e}")

            access_token = parsed.get("access_token")
            refresh_token = parsed.get("refresh_token")
            email = parsed.get("email")

            expires_at = parsed.get("expires_at")
            expires_in = parsed.get("expires_in")
            expiry_date = None
            if isinstance(expires_at, (int, float)):
                expiry_date = float(expires_at)
            elif isinstance(expires_in, (int, float)):
                expiry_date = time.time() + float(expires_in)
        else:
            access_token = raw_entry

        if not access_token:
            raise ValueError("access_token is required")

        if not refresh_token:
            refresh_token = (
                Prompt.ask("Enter refresh_token", default="").strip() or None
            )
        if not refresh_token:
            raise ValueError("refresh_token is required for ChatGPT credential setup")

        token_payload = self._decode_jwt_payload(access_token)
        account_id = self._extract_account_id(access_token, token_payload)

        if not account_id:
            account_id = (
                Prompt.ask(
                    "Could not auto-detect account_id. Enter account ID", default=""
                ).strip()
                or None
            )
        if not account_id:
            raise ValueError("account_id is required")

        if expiry_date is None:
            token_expiry = self._extract_expiry_from_access_token(access_token)
            expiry_date = token_expiry

        if expiry_date is None:
            expiry_input = Prompt.ask(
                "Enter expiry_date (unix timestamp seconds)",
                default=str(int(time.time() + 1800)),
            ).strip()
            try:
                expiry_date = float(expiry_input)
            except ValueError as e:
                raise ValueError(f"Invalid expiry_date: {e}")

        inferred_email = self._extract_email(token_payload)
        email = (email or inferred_email or "").strip() or None
        if not email:
            email = (
                Prompt.ask(
                    "Enter email (for credential identification)", default=""
                ).strip()
                or None
            )

        creds["access_token"] = access_token
        creds["refresh_token"] = refresh_token
        creds["expiry_date"] = float(expiry_date)
        creds["account_id"] = account_id
        creds["_proxy_metadata"] = {
            **creds.get("_proxy_metadata", {}),
            "email": email,
            "last_check_timestamp": time.time(),
        }

        if path:
            if not await self._save_credentials(path, creds):
                raise IOError(
                    f"Failed to save ChatGPT OAuth credentials for '{display_name}'."
                )

        return creds

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            reason = ""
            if force_interactive:
                reason = "interactive re-auth explicitly requested"
            elif not creds.get("access_token"):
                reason = "access token is missing"
            elif not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                if reason == "token is expired" and creds.get("refresh_token") and path:
                    try:
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(
                            f"ChatGPT token refresh failed for '{display_name}': {e}. Falling back to manual OAuth."
                        )

                coordinator = get_reauth_coordinator()

                async def _do_interactive_oauth():
                    return await self._perform_interactive_oauth(
                        path, creds, display_name
                    )

                return await coordinator.execute_reauth(
                    credential_path=path or display_name,
                    provider_name=self.ENV_PREFIX,
                    reauth_func=_do_interactive_oauth,
                    timeout=float(self._reauth_timeout_seconds),
                )

            return creds
        except Exception as e:
            raise ValueError(
                f"Failed to initialize ChatGPT OAuth for '{display_name}': {e}"
            )

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def proactively_refresh(self, credential_identifier: str):
        try:
            creds = await self._load_credentials(credential_identifier)
        except IOError:
            return

        if self._is_token_expired(creds):
            await self._queue_refresh(
                credential_identifier, force=False, needs_reauth=False
            )

    def _get_provider_file_prefix(self) -> str:
        return "openai_chatgpt"

    def _get_oauth_base_dir(self) -> Path:
        return Path.cwd() / "oauth_creds"

    def _find_existing_credential_by_email(
        self, email: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        for cred_file in glob(pattern):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                existing_email = creds.get("_proxy_metadata", {}).get("email")
                if existing_email == email:
                    return Path(cred_file)
            except (json.JSONDecodeError, IOError):
                continue

        return None

    def _get_next_credential_number(self, base_dir: Optional[Path] = None) -> int:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        existing_numbers = []
        for cred_file in glob(pattern):
            match = re.search(r"_oauth_(\d+)\.json$", cred_file)
            if match:
                existing_numbers.append(int(match.group(1)))

        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    def _build_credential_path(
        self, base_dir: Optional[Path] = None, number: Optional[int] = None
    ) -> Path:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        if number is None:
            number = self._get_next_credential_number(base_dir)

        prefix = self._get_provider_file_prefix()
        return base_dir / f"{prefix}_oauth_{number}.json"

    async def setup_credential(
        self, base_dir: Optional[Path] = None
    ) -> OpenAIChatGPTCredentialSetupResult:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        base_dir.mkdir(exist_ok=True)

        try:
            temp_creds = {
                "_proxy_metadata": {
                    "display_name": "new OpenAI ChatGPT credential",
                }
            }
            new_creds = await self.initialize_token(temp_creds)

            email = new_creds.get("_proxy_metadata", {}).get("email")
            account_id = new_creds.get("account_id")
            if not email:
                return OpenAIChatGPTCredentialSetupResult(
                    success=False,
                    error="Could not retrieve email from ChatGPT token",
                )

            existing_path = self._find_existing_credential_by_email(email, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
            else:
                file_path = self._build_credential_path(base_dir)

            if not await self._save_credentials(str(file_path), new_creds):
                return OpenAIChatGPTCredentialSetupResult(
                    success=False,
                    error=f"Failed to save credentials to disk at {file_path.name}",
                )

            return OpenAIChatGPTCredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=email,
                account_id=account_id,
                is_update=is_update,
                credentials=new_creds,
            )
        except Exception as e:
            lib_logger.error(f"ChatGPT credential setup failed: {e}")
            return OpenAIChatGPTCredentialSetupResult(success=False, error=str(e))

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        email = creds.get("_proxy_metadata", {}).get("email", "unknown")
        prefix = f"{self.ENV_PREFIX}_{cred_number}"

        lines = [
            f"# {self.ENV_PREFIX} Credential #{cred_number} for: {email}",
            f"# Exported from: {self._get_provider_file_prefix()}_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#",
            "# To combine multiple credentials into one .env file, copy these lines",
            "# and ensure each credential has a unique number (1, 2, 3, etc.)",
            "",
            f"{prefix}_ACCESS_TOKEN={creds.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={creds.get('refresh_token', '')}",
            f"{prefix}_EXPIRES_AT={creds.get('expiry_date', 0)}",
            f"{prefix}_EXPIRY_DATE={creds.get('expiry_date', 0)}",
            f"{prefix}_ACCOUNT_ID={creds.get('account_id', '')}",
            f"{prefix}_EMAIL={email}",
        ]

        return lines

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        credentials = []
        for cred_file in sorted(glob(pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})
                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                credentials.append(
                    {
                        "file_path": cred_file,
                        "email": metadata.get("email", "unknown"),
                        "account_id": creds.get("account_id"),
                        "number": number,
                    }
                )
            except Exception:
                continue

        return credentials

    def delete_credential(self, credential_path: str) -> bool:
        try:
            cred_path = Path(credential_path)

            prefix = self._get_provider_file_prefix()
            if not cred_path.name.startswith(f"{prefix}_oauth_"):
                lib_logger.error(
                    f"File {cred_path.name} does not appear to be a {self.ENV_PREFIX} credential"
                )
                return False

            if not cred_path.exists():
                lib_logger.warning(f"Credential file does not exist: {credential_path}")
                return False

            self._credentials_cache.pop(credential_path, None)
            cred_path.unlink()
            return True
        except Exception as e:
            lib_logger.error(f"Failed to delete ChatGPT credential: {e}")
            return False

    async def get_account_id(self, credential_identifier: str) -> Optional[str]:
        creds = await self._load_credentials(credential_identifier)
        account_id = creds.get("account_id")
        if account_id:
            return str(account_id)

        access_token = creds.get("access_token")
        if isinstance(access_token, str):
            payload = self._decode_jwt_payload(access_token)
            account_id = self._extract_account_id(access_token, payload)
            if account_id:
                creds["account_id"] = account_id
                if credential_identifier.startswith("env://"):
                    self._credentials_cache[credential_identifier] = creds
                else:
                    await self._save_credentials(credential_identifier, creds)
                return account_id

        return None
