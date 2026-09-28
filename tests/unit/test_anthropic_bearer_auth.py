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

"""Tests for Bearer auth on the standard Anthropic Messages API."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_anthropic import ChatAnthropic

from skillspector.providers import registry
from skillspector.providers.anthropic import AnthropicProvider
from skillspector.providers.anthropic.provider import (
    _BearerTransport,
    _ChatAnthropicBearer,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_SCHEME", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_SSL_VERIFY", raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


class TestAnthropicBearerChatModel:
    def test_default_uses_stock_chat_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        llm = AnthropicProvider().create_chat_model("claude-sonnet-4-6", max_tokens=1024)
        assert type(llm) is ChatAnthropic

    def test_bearer_scheme_uses_subclass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "gateway-token")
        monkeypatch.setenv("ANTHROPIC_AUTH_SCHEME", "bearer")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")
        llm = AnthropicProvider().create_chat_model("claude-sonnet-4-6", max_tokens=1024)
        assert isinstance(llm, _ChatAnthropicBearer)
        assert str(llm.anthropic_api_url).rstrip("/") == "https://gateway.example.com"


class TestBearerTransport:
    def test_replaces_x_api_key_with_bearer_and_preserves_url_and_body(self) -> None:
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        original = httpx.Request(
            method="POST",
            url="https://gateway.example.com/v1/messages",
            headers={
                "x-api-key": "sk-ant-secret",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            content=json.dumps(body).encode("utf-8"),
        )
        transport = _BearerTransport("my-bearer", verify=False)
        captured: list[httpx.Request] = []

        class _CapturingTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(request)
                return httpx.Response(status_code=200, json={"type": "message"})

        transport._inner = _CapturingTransport()
        transport.handle_request(original)

        assert len(captured) == 1
        rewritten = captured[0]
        assert str(rewritten.url) == "https://gateway.example.com/v1/messages"
        assert json.loads(rewritten.content) == body
        assert rewritten.headers["authorization"] == "Bearer my-bearer"
        assert "x-api-key" not in rewritten.headers
        assert rewritten.headers["anthropic-version"] == "2023-06-01"
