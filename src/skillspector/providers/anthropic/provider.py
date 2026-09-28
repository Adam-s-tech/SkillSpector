# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic provider — Claude models via api.anthropic.com.

Reads ``ANTHROPIC_API_KEY`` for credentials and honors ``ANTHROPIC_BASE_URL``
as an explicit endpoint override (e.g. a local proxy); when unset, requests
go to api.anthropic.com. Constructs ``langchain_anthropic.ChatAnthropic``
directly. It defaults to Opus 4.6 for analyzers and Sonnet 4.6 for
``meta_analyzer`` (cheaper for the high-volume filter pass).

Set ``ANTHROPIC_AUTH_SCHEME=bearer`` when the endpoint expects
``Authorization: Bearer`` (common on corporate LLM gateways) instead of
Anthropic's default ``x-api-key`` header. The token is read from
``ANTHROPIC_API_KEY``; request bodies and URLs stay on the standard Messages
API (unlike ``anthropic_proxy``, which targets Vertex raw-predict).
"""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from skillspector.inference_usage import (
    register_chat_model_controls,
    retained_chat_model_controls,
)
from skillspector.providers import registry
from skillspector.providers.chat_models import resolve_reasoning_effort, resolve_sampling_parameters
from skillspector.providers.structured_output import rejects_forced_tool_call

# Default endpoint; overridden by ``ANTHROPIC_BASE_URL`` when set.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


def _use_bearer_auth() -> bool:
    return os.environ.get("ANTHROPIC_AUTH_SCHEME", "").strip().lower() == "bearer"


def _ssl_verify() -> bool | str:
    """Return the SSL verification setting from ``SKILLSPECTOR_SSL_VERIFY``."""
    val = os.environ.get("SKILLSPECTOR_SSL_VERIFY", "").strip().lower()
    if val == "false":
        return False
    if val and val != "true":
        return val
    return True


def _apply_bearer_auth(request: httpx.Request, bearer_token: str) -> httpx.Request:
    """Replace ``x-api-key`` with ``Authorization: Bearer``; keep URL and body."""
    headers = httpx.Headers({k: v for k, v in request.headers.items() if k.lower() != "x-api-key"})
    headers["authorization"] = f"Bearer {bearer_token}"
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=headers,
        content=request.content,
    )


class _BearerTransport(httpx.BaseTransport):
    """Sync httpx transport that swaps Anthropic API-key auth for Bearer tokens."""

    def __init__(self, bearer_token: str, *, verify: bool | str = True):
        self._bearer_token = bearer_token
        self._inner = httpx.HTTPTransport(verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._inner.handle_request(_apply_bearer_auth(request, self._bearer_token))

    def close(self) -> None:
        self._inner.close()


class _BearerAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that swaps Anthropic API-key auth for Bearer tokens."""

    def __init__(self, bearer_token: str, *, verify: bool | str = True):
        self._bearer_token = bearer_token
        self._inner = httpx.AsyncHTTPTransport(verify=verify)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(
            _apply_bearer_auth(request, self._bearer_token)
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


class _ChatAnthropicBearer(ChatAnthropic):
    """``ChatAnthropic`` subclass that sends Bearer auth on the Messages API."""

    _bearer_token: str = ""
    _bearer_ssl_verify: bool | str = True

    def __init__(self, *, bearer_token: str, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "_bearer_token", bearer_token)
        object.__setattr__(self, "_bearer_ssl_verify", _ssl_verify())

    @cached_property
    def _client(self) -> anthropic.Client:  # type: ignore[override]
        params = self._client_params
        transport = _BearerTransport(self._bearer_token, verify=self._bearer_ssl_verify)
        http_client = httpx.Client(transport=transport)
        return anthropic.Client(
            api_key=params["api_key"],
            base_url=params.get("base_url"),
            max_retries=params.get("max_retries", 2),
            default_headers=params.get("default_headers"),
            timeout=params.get("timeout"),
            http_client=http_client,
        )

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:  # type: ignore[override]
        params = self._client_params
        transport = _BearerAsyncTransport(self._bearer_token, verify=self._bearer_ssl_verify)
        http_client = httpx.AsyncClient(transport=transport)
        return anthropic.AsyncClient(
            api_key=params["api_key"],
            base_url=params.get("base_url"),
            max_retries=params.get("max_retries", 2),
            default_headers=params.get("default_headers"),
            timeout=params.get("timeout"),
            http_client=http_client,
        )


class AnthropicProvider:
    """Anthropic credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL = "claude-opus-4-6"
    SLOT_DEFAULTS: dict[str, str] = {
        "meta_analyzer": "claude-sonnet-4-6",
    }

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, base_url)`` from ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip() or None
        return api_key, base_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatAnthropic`` using native Anthropic credentials."""
        creds = self.resolve_credentials()
        if creds is None:
            return None

        api_key, base_url = creds
        bearer_auth = _use_bearer_auth()
        kwargs = {
            "model_name": model,
            "api_key": SecretStr("anthropic-bearer-placeholder" if bearer_auth else api_key),
            "base_url": base_url or ANTHROPIC_BASE_URL,
            "max_tokens_to_sample": max_tokens,
            "timeout": timeout,
            "stop": None,
        }
        effort = resolve_reasoning_effort()
        if effort is not None:
            kwargs["effort"] = effort
        sampling_parameters = resolve_sampling_parameters()
        kwargs.update(sampling_parameters)
        if bearer_auth:
            chat_model = _ChatAnthropicBearer(bearer_token=api_key, **kwargs)
        else:
            chat_model = ChatAnthropic(**kwargs)
        register_chat_model_controls(
            chat_model,
            retained_chat_model_controls(
                chat_model,
                ("temperature", "reasoning_effort"),
            ),
            requested_controls={
                "temperature": sampling_parameters.get("temperature"),
                "reasoning_effort": effort,
            },
        )
        return chat_model

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL

    def structured_output_method(self, model: str) -> str | None:
        """``with_structured_output`` method for *model*: registry entry, then family prefix, else ``None``."""
        declared = registry.lookup_structured_output_method(REGISTRY_PATH, model)
        if declared:
            return declared
        return "json_schema" if rejects_forced_tool_call(model) else None
