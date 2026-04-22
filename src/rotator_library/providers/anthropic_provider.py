# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/anthropic_provider.py

import copy
import hashlib
import json
import os
import re
import time
import logging
import uuid
from typing import Union, AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path

import httpx
import litellm
from litellm.exceptions import RateLimitError

from .provider_interface import ProviderInterface, QuotaGroupMap, UsageResetConfigDef
from .anthropic_auth_base import AnthropicAuthBase
from .provider_cache import create_provider_cache
from .utilities.anthropic_quota_tracker import AnthropicQuotaTracker
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CLAUDE CODE IMPERSONATION CONSTANTS
# =============================================================================

CLAUDE_CODE_VERSION = "2.1.42"
TOOL_PREFIX = "mcp_"

ANTHROPIC_API_BASE = "https://api.anthropic.com"

BETA_CLAUDE_CODE = "claude-code-20250219"
BETA_OAUTH = "oauth-2025-04-20"
BETA_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
BETA_EFFORT = "effort-2025-11-24"
BETA_CONTEXT_MANAGEMENT = "context-management-2025-06-27"
BETA_FAST_MODE = "fast-mode-2026-02-01"

CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Fallback model list — only used if live fetch fails
FALLBACK_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
]

# Stop reason mapping: Anthropic -> OpenAI
STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "pause_turn": "stop",
}

# Lazy-initialised server-side cache for thinking block signatures.
# Allows us to re-attach signatures when OpenAI-format clients send back
# reasoning_content without the signature (which they can't preserve).
_thinking_sig_cache = None


def _get_thinking_cache():
    global _thinking_sig_cache
    if _thinking_sig_cache is None:
        _thinking_sig_cache = create_provider_cache(
            "anthropic_thinking_signatures",
            memory_ttl_seconds=7200,  # 2 hours in memory
            disk_ttl_seconds=172800,  # 48 hours on disk
        )
    return _thinking_sig_cache


def _coerce_openai_content_part(part: Any) -> Optional[Dict[str, Any]]:
    """Normalize OpenAI content parts into dict form for Anthropic translation."""
    if isinstance(part, dict):
        return part
    if isinstance(part, str):
        text = part.strip()
        if text:
            return {"type": "text", "text": text}
    return None


def _anthropic_tool_name(name: str) -> str:
    """Normalize client tool names into a Claude-Code-like PascalCase shape."""
    if not name:
        return name

    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    parts = [part for part in re.split(r"[^A-Za-z0-9]+|\s+", spaced) if part]
    if not parts:
        return name[:1].upper() + name[1:]

    return "".join(part[:1].upper() + part[1:] for part in parts)


def _rewrite_opencode_environment_metadata(text: str) -> str:
    """Rewrite OpenCode env metadata into plain prose to avoid client fingerprinting."""
    env_pattern = re.compile(
        r"Here is some useful information about the environment you are running in:\n"
        r"<env>\n(?P<env_body>.*?)\n</env>\n<directories>\n(?P<dirs_body>.*?)\n</directories>",
        flags=re.DOTALL,
    )

    def replace_match(match: re.Match[str]) -> str:
        env_body = match.group("env_body")
        dirs_body = match.group("dirs_body")

        fields: Dict[str, str] = {}
        for raw_line in env_body.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()

        summary_parts = []
        platform = fields.get("Platform")
        git_repo = fields.get("Is directory a git repo")
        working_dir = fields.get("Working directory")
        workspace_root = fields.get("Workspace root folder")
        current_date = fields.get("Today's date")

        if platform:
            summary_parts.append(f"platform {platform}")
        if git_repo:
            repo_text = (
                "inside a git repo"
                if git_repo.lower() == "yes"
                else "not in a git repo"
            )
            summary_parts.append(repo_text)
        if working_dir:
            summary_parts.append(f"working directory {working_dir}")
        if workspace_root:
            summary_parts.append(f"workspace root {workspace_root}")
        if current_date:
            summary_parts.append(f"date {current_date}")

        summary = "Environment summary: "
        if summary_parts:
            summary += ", ".join(summary_parts) + "."
        else:
            summary += "local workspace context available."

        directories_text = dirs_body.strip()
        if directories_text:
            summary += f" Directories: {directories_text}."
        else:
            summary += " Directories: none listed."

        return summary

    return env_pattern.sub(replace_match, text)


class AnthropicProvider(AnthropicAuthBase, AnthropicQuotaTracker, ProviderInterface):
    """
    Anthropic provider using OAuth authentication (Claude Pro/Max).
    Calls Anthropic's Messages API directly, impersonating Claude Code.
    """

    skip_cost_calculation = True

    # =========================================================================
    # ROTATION & TIER CONFIGURATION
    # =========================================================================

    # Sequential mode preserves Anthropic prompt caching (per-credential)
    default_rotation_mode: str = "sequential"

    # Provider name for env var lookups (ROTATION_MODE_ANTHROPIC, etc.)
    provider_env_name: str = "anthropic"

    # Priority tiers for credential ordering (lower number = used first).
    # get_credential_tier_name() maps credentials to these via env vars.
    tier_priorities = {"priority-1": 1, "priority-2": 2}
    default_tier_priority: int = 999

    # Fair cycle conflicts with priority rotation: it blocks the primary
    # credential after exhaustion until ALL credentials exhaust, defeating
    # the "return to primary when cooldown expires" behavior.
    # Override with FAIR_CYCLE_ANTHROPIC=true if needed.
    default_fair_cycle_enabled = False

    usage_reset_configs = {
        frozenset({1, 2}): UsageResetConfigDef(
            window_seconds=5 * 60 * 60,
            mode="per_model",
            description="5-hour Anthropic quota window",
            field_name="models",
        ),
        "default": UsageResetConfigDef(
            window_seconds=7 * 24 * 60 * 60,
            mode="per_model",
            description="7-day Anthropic quota window",
            field_name="models",
        ),
    }

    model_quota_groups: QuotaGroupMap = {
        "5h-limit": ["_5h_window"],
        "7d-limit": ["_7d_window"],
    }

    def __init__(self):
        super().__init__()
        self._init_quota_tracker()

    def has_custom_logic(self) -> bool:
        return True

    # =========================================================================
    # CREDENTIAL PRIORITY (env-var based)
    # =========================================================================

    @staticmethod
    def _extract_credential_number(credential: str) -> Optional[int]:
        """
        Extract the numeric index from a credential identifier.

        Handles:
        - File paths: /app/oauth_creds/anthropic_oauth_2.json -> 2
        - Env URIs:   env://anthropic/2 -> 2
        """
        if not credential:
            return None

        # env://anthropic/2
        env_match = re.match(r"^env://[^/]+/(\d+)$", credential)
        if env_match:
            return int(env_match.group(1))

        # /app/oauth_creds/anthropic_oauth_2.json
        file_match = re.search(r"_oauth_(\d+)\.json$", credential)
        if file_match:
            return int(file_match.group(1))

        return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Map credential -> synthetic tier name "priority-{N}".

        Priority is configured via env vars:
            ANTHROPIC_CREDENTIAL_PRIORITY_1=2   (oauth_1 -> priority 2)
            ANTHROPIC_CREDENTIAL_PRIORITY_2=1   (oauth_2 -> priority 1, used first)

        Falls back to default_tier_priority when not configured.
        """
        number = self._extract_credential_number(credential)
        if number is not None:
            raw = os.getenv(f"ANTHROPIC_CREDENTIAL_PRIORITY_{number}")
            if raw is not None:
                try:
                    priority = int(raw)
                    if priority >= 1:
                        tier_name = f"priority-{priority}"
                        # Register tier if not pre-defined (e.g. priority-3+)
                        self.tier_priorities.setdefault(tier_name, priority)
                        return tier_name
                except ValueError:
                    lib_logger.warning(
                        f"Invalid ANTHROPIC_CREDENTIAL_PRIORITY_{number}={raw!r}; "
                        f"using default priority {self.default_tier_priority}"
                    )

        return f"priority-{self.default_tier_priority}"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch models live from Anthropic API, falling back to hardcoded list."""
        try:
            access_token = await self.get_access_token(api_key)
            headers = self._build_anthropic_headers(access_token)
            resp = await client.get(
                f"{ANTHROPIC_API_BASE}/v1/models",
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    f"anthropic/{m['id']}" for m in data.get("data", []) if m.get("id")
                ]
                if models:
                    return models
        except Exception as e:
            lib_logger.debug(f"Failed to fetch Anthropic models live: {e}")

        return [f"anthropic/{m}" for m in FALLBACK_MODELS]

    # =========================================================================
    # OPENAI -> ANTHROPIC MESSAGE CONVERSION
    # =========================================================================

    def _openai_messages_to_anthropic(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI-format messages to Anthropic Messages API format.
        Returns (system_blocks, anthropic_messages).
        """
        system_blocks = []
        anthropic_messages = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for raw_block in content:
                        block = _coerce_openai_content_part(raw_block)
                        if block and block.get("type") == "text":
                            system_blocks.append(
                                {"type": "text", "text": block.get("text", "")}
                            )
                continue

            if role == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content
                    if isinstance(content, str)
                    else json.dumps(content),
                }
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    if isinstance(anthropic_messages[-1]["content"], list):
                        anthropic_messages[-1]["content"].append(tool_result)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {
                                "type": "text",
                                "text": anthropic_messages[-1]["content"],
                            },
                            tool_result,
                        ]
                else:
                    anthropic_messages.append(
                        {"role": "user", "content": [tool_result]}
                    )
                continue

            if role == "assistant":
                blocks = []

                reasoning = msg.get("reasoning_content")
                if reasoning:
                    # Try server-side cache first (signature preserved from
                    # the original Anthropic response)
                    cached = self._retrieve_thinking_blocks(reasoning)
                    if cached:
                        lib_logger.info(
                            f"Thinking signature cache HIT – restored {len(cached)} block(s)"
                        )
                        blocks.extend(cached)
                    else:
                        # Fallback: inline signature from client (custom clients)
                        thinking_sig = msg.get("thinking_signature")
                        if thinking_sig and len(thinking_sig) >= 100:
                            lib_logger.debug(
                                "Using inline thinking signature from client"
                            )
                            blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": reasoning,
                                    "signature": thinking_sig,
                                }
                            )
                        else:
                            lib_logger.warning(
                                "Thinking signature cache MISS – dropping thinking block"
                            )

                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for raw_block in content:
                        block = _coerce_openai_content_part(raw_block)
                        if not block:
                            continue
                        if (
                            block.get("type") == "text"
                            and block.get("text", "").strip()
                        ):
                            blocks.append({"type": "text", "text": block["text"]})
                        elif block.get("type") == "image_url":
                            image_url = block.get("image_url", {})
                            if isinstance(image_url, str):
                                url = image_url
                            elif isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = ""
                            if url.startswith("data:"):
                                parts = url.split(",", 1)
                                media_type = (
                                    parts[0].replace("data:", "").replace(";base64", "")
                                )
                                blocks.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": parts[1] if len(parts) > 1 else "",
                                        },
                                    }
                                )

                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        input_data = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": _anthropic_tool_name(func.get("name", "")),
                            "input": input_data,
                        }
                    )

                if blocks:
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                continue

            # User messages
            if isinstance(content, str):
                if content.strip():
                    anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = []
                for raw_block in content:
                    block = _coerce_openai_content_part(raw_block)
                    if not block:
                        continue
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {})
                        if isinstance(image_url, str):
                            url = image_url
                        elif isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = ""
                        if url.startswith("data:"):
                            parts = url.split(",", 1)
                            media_type = (
                                parts[0].replace("data:", "").replace(";base64", "")
                            )
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": parts[1] if len(parts) > 1 else "",
                                    },
                                }
                            )
                if blocks:
                    anthropic_messages.append({"role": "user", "content": blocks})

        # Enforce Anthropic message constraints:
        # 1. Merge consecutive same-role messages
        # 2. Ensure conversation starts with user
        # 3. Ensure conversation doesn't end with trailing assistant (prefill)
        anthropic_messages = self._enforce_alternation(anthropic_messages)

        return system_blocks, anthropic_messages

    @staticmethod
    def _enforce_alternation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enforce strict user/assistant alternation required by Anthropic API.

        Merges consecutive same-role messages and ensures the conversation
        starts with a user message.
        """
        if not messages:
            return messages

        # Merge consecutive same-role messages
        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Same role — merge content into previous message
                prev_content = merged[-1]["content"]
                new_content = msg["content"]
                # Normalize both to list form
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(new_content, str):
                    new_content = [{"type": "text", "text": new_content}]
                if not isinstance(prev_content, list):
                    prev_content = [prev_content]
                if not isinstance(new_content, list):
                    new_content = [new_content]
                merged[-1]["content"] = prev_content + new_content
            else:
                merged.append(msg)

        # Ensure conversation starts with user (Anthropic requirement)
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "Continue."})

        return merged

    def _retrieve_thinking_blocks(
        self, reasoning_content: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Look up cached thinking blocks with signatures for given thinking content."""
        cache_key = hashlib.sha256(reasoning_content.encode()).hexdigest()
        cached = _get_thinking_cache().retrieve(cache_key)
        if not cached:
            return None
        try:
            blocks_data = json.loads(cached)
            result = [
                {
                    "type": "thinking",
                    "thinking": b["thinking"],
                    "signature": b["signature"],
                }
                for b in blocks_data
                if b.get("signature")
            ]
            return result if result else None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _openai_tools_to_anthropic(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        result = []
        for tool in tools:
            func = tool.get("function", {})
            schema = func.get("parameters", {"type": "object"})
            result.append(
                {
                    "name": _anthropic_tool_name(func.get("name", "")),
                    "description": func.get("description", ""),
                    "input_schema": schema,
                }
            )
        return result

    # =========================================================================
    # MCP_ TOOL NAME PREFIXING / STRIPPING
    # =========================================================================

    def _prefix_tool_names(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add mcp_ prefix to tool names in definitions, messages, and tool_choice."""
        payload = copy.deepcopy(payload)

        if payload.get("tools"):
            for tool in payload["tools"]:
                if tool.get("name"):
                    tool["name"] = f"{TOOL_PREFIX}{tool['name']}"

        if payload.get("messages"):
            for msg in payload["messages"]:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            if block.get("name"):
                                block["name"] = f"{TOOL_PREFIX}{block['name']}"

        # Also prefix tool_choice.name to match prefixed tool definitions
        tc = payload.get("tool_choice")
        if isinstance(tc, dict) and tc.get("type") == "tool" and tc.get("name"):
            tc["name"] = f"{TOOL_PREFIX}{tc['name']}"

        return payload

    def _strip_tool_prefix(self, name: str) -> str:
        """Remove mcp_ prefix and restore client-facing tool casing."""
        if name and name.startswith(TOOL_PREFIX):
            stripped = name[len(TOOL_PREFIX) :]
            if stripped:
                return stripped[:1].lower() + stripped[1:]
            return stripped
        return name

    # =========================================================================
    # SYSTEM PROMPT HANDLING
    # =========================================================================

    def _inject_system_prompt(
        self, system_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepend Claude Code identity to system prompt.

        The claude-code beta validates that the first system block is the
        exact prefix string byte-for-byte.  Merging client text into it
        (even with ``\\n\\n``) triggers a 400.  Keep them as separate blocks.
        """
        result = [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}]
        if system_blocks:
            for block in system_blocks:
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    rewritten = copy.deepcopy(block)
                    rewritten_text = re.sub(
                        r"OpenCode", "Claude Code", text, flags=re.IGNORECASE
                    )
                    rewritten["text"] = _rewrite_opencode_environment_metadata(
                        rewritten_text
                    )
                    result.append(rewritten)
        return result

    # =========================================================================
    # PROMPT CACHING
    # =========================================================================

    def _inject_cache_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject cache_control breakpoints into the Anthropic payload to enable
        prompt caching. Marks the last system block, last tool, and the last
        content block of the final message for caching.

        Anthropic caches the full prefix up to each breakpoint. This saves
        ~90% on cached input token costs and reduces latency.
        """
        system_cache_marker = {"type": "ephemeral", "ttl": "1h"}
        cache_marker = {"type": "ephemeral", "ttl": "1h"}

        # 1. Cache the last system block (system prompt rarely changes)
        system = payload.get("system")
        if system and isinstance(system, list) and len(system) > 0:
            system[-1]["cache_control"] = system_cache_marker

        # 2. Cache the last tool definition (tools rarely change)
        tools = payload.get("tools")
        if tools and isinstance(tools, list) and len(tools) > 0:
            tools[-1]["cache_control"] = cache_marker

        # 3. Cache the end of conversation history for multi-turn caching.
        #    Mark the last content block of the latest message so the full
        #    conversation, including the newest turn, is cached across turns.
        messages = payload.get("messages")
        if messages and len(messages) >= 1:
            last_msg = messages[-1]
            content = last_msg.get("content")
            if isinstance(content, list) and len(content) > 0:
                last_block = content[-1]
                if isinstance(last_block, dict):
                    last_block["cache_control"] = cache_marker
                else:
                    normalized = _coerce_openai_content_part(last_block)
                    if normalized is not None:
                        normalized["cache_control"] = cache_marker
                        content[-1] = normalized
            elif isinstance(content, str) and content:
                # Convert string content to block format so we can attach cache_control
                last_msg["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_marker}
                ]

        return payload

    # =========================================================================
    # ANTHROPIC SSE -> OPENAI CHUNK CONVERSION
    # =========================================================================

    def _anthropic_event_to_openai_chunks(
        self,
        event_type: str,
        data: Dict[str, Any],
        model_id: str,
        stream_state: Dict[str, Any],
    ):
        """
        Convert a single Anthropic SSE event to OpenAI-format chunk(s).
        Yields litellm.ModelResponse-compatible dicts.
        """
        if event_type == "message_start":
            message = data.get("message", {})
            usage = message.get("usage", {})
            stream_state["input_tokens"] = usage.get("input_tokens", 0)
            stream_state["message_id"] = message.get(
                "id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
            )
            return

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            block_type = block.get("type")
            index = data.get("index", 0)
            stream_state["current_block_type"] = block_type
            stream_state["current_block_index"] = index

            if block_type == "redacted_thinking":
                # Redacted thinking blocks don't stream visible content.
                return

            if block_type == "thinking":
                stream_state["_block_thinking"] = ""
                stream_state["_block_signature"] = ""

            if block_type == "tool_use":
                tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                raw_name = block.get("name", "")
                name = self._strip_tool_prefix(raw_name)
                stream_state.setdefault("tool_calls", {})
                stream_state["tool_calls"][index] = {
                    "id": tool_id,
                    "name": name,
                    "arguments": "",
                    "tc_index": len(stream_state["tool_calls"]),
                }
                stream_state["has_tool_calls"] = True
                yield {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": stream_state["tool_calls"][index][
                                            "tc_index"
                                        ],
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                    "model": model_id,
                    "object": "chat.completion.chunk",
                    "id": stream_state.get(
                        "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    ),
                    "created": int(time.time()),
                }
            return

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    stream_state["accumulated_text"] = (
                        stream_state.get("accumulated_text", "") + text
                    )
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "id": stream_state.get(
                            "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        ),
                        "created": int(time.time()),
                    }

            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking", "")
                if thinking:
                    stream_state["accumulated_thinking"] = (
                        stream_state.get("accumulated_thinking", "") + thinking
                    )
                    stream_state["_block_thinking"] = (
                        stream_state.get("_block_thinking", "") + thinking
                    )
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": thinking},
                                "finish_reason": None,
                            }
                        ],
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "id": stream_state.get(
                            "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        ),
                        "created": int(time.time()),
                    }

            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                block_index = data.get("index", 0)
                tc_info = stream_state.get("tool_calls", {}).get(block_index)
                if tc_info and partial:
                    tc_info["arguments"] += partial
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tc_info["tc_index"],
                                            "function": {"arguments": partial},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "id": stream_state.get(
                            "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        ),
                        "created": int(time.time()),
                    }

            elif delta_type == "signature_delta":
                sig = delta.get("signature", "")
                stream_state["thinking_signature"] = (
                    stream_state.get("thinking_signature", "") + sig
                )
                stream_state["_block_signature"] = (
                    stream_state.get("_block_signature", "") + sig
                )

            return

        if event_type == "content_block_stop":
            if stream_state.get("current_block_type") == "thinking":
                block_thinking = stream_state.pop("_block_thinking", "")
                block_sig = stream_state.pop("_block_signature", "")
                if block_thinking and block_sig:
                    stream_state.setdefault("_thinking_blocks", []).append(
                        {
                            "thinking": block_thinking,
                            "signature": block_sig,
                        }
                    )
            return

        if event_type == "message_delta":
            delta = data.get("delta", {})
            usage = data.get("usage", {})
            stop_reason = delta.get("stop_reason", "end_turn")
            finish_reason = STOP_REASON_MAP.get(stop_reason, "stop")
            stream_state["finish_reason"] = finish_reason

            output_tokens = usage.get("output_tokens", 0)
            input_tokens = stream_state.get("input_tokens", 0)

            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": stream_state.get(
                    "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                ),
                "created": int(time.time()),
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }

            # Cache thinking blocks with signatures for multi-turn preservation
            thinking_blocks = stream_state.get("_thinking_blocks")
            if thinking_blocks:
                full_thinking = "".join(b["thinking"] for b in thinking_blocks)
                cache_key = hashlib.sha256(full_thinking.encode()).hexdigest()
                _get_thinking_cache().store(cache_key, json.dumps(thinking_blocks))
                lib_logger.info(
                    f"Thinking signature cache STORE – {len(thinking_blocks)} block(s), key={cache_key[:12]}..."
                )

            return

    # =========================================================================
    # MAIN API CALL
    # =========================================================================

    def _build_beta_header(self, payload: Optional[Dict[str, Any]] = None) -> str:
        betas = [
            BETA_CLAUDE_CODE,
            BETA_OAUTH,
            BETA_INTERLEAVED_THINKING,
        ]

        if payload is None:
            return ",".join(betas)

        # Always send effort beta for adaptive-thinking models (upstream behavior)
        model = payload.get("model", "")
        if self._model_supports_adaptive_thinking(model):
            betas.append(BETA_EFFORT)

        if "context_management" in payload:
            betas.append(BETA_CONTEXT_MANAGEMENT)

        if "speed" in payload:
            betas.append(BETA_FAST_MODE)

        return ",".join(betas)

    @staticmethod
    def _build_claude_session_id(
        payload: Optional[Dict[str, Any]] = None,
        transaction_context: Optional[Any] = None,
    ) -> str:
        if transaction_context is not None:
            request_id = getattr(transaction_context, "request_id", None)
            if request_id:
                return request_id

        if payload:
            messages = payload.get("messages") or []
            if messages:
                material = json.dumps(messages, sort_keys=True, default=str)
                digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
                return str(uuid.UUID(digest[:32]))

        return str(uuid.uuid4())

    def _build_anthropic_headers(
        self,
        access_token: str,
        payload: Optional[Dict[str, Any]] = None,
        transaction_context: Optional[Any] = None,
    ) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": self._build_beta_header(payload),
            "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
            "x-app": "cli",
            "X-Claude-Code-Session-Id": self._build_claude_session_id(
                payload, transaction_context
            ),
            "anthropic-dangerous-direct-browser-access": "true",
        }

    @staticmethod
    def _model_supports_adaptive_thinking(model: str) -> bool:
        """Models trained on adaptive thinking (4.6+)."""
        model_lower = model.lower()
        # Match known adaptive-thinking families while allowing newer minor releases.
        return bool(re.search(r"(?:opus|sonnet|haiku)-4-[6-9](?:\D|$)", model_lower))

    def _build_anthropic_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the Anthropic Messages API payload from OpenAI-format kwargs."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "")

        if "/" in model:
            model = model.split("/", 1)[1]

        system_blocks, anthropic_messages = self._openai_messages_to_anthropic(messages)
        system_blocks = self._inject_system_prompt(system_blocks)

        tools = self._openai_tools_to_anthropic(kwargs.get("tools"))

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 16384),
            "stream": True,
        }

        if system_blocks:
            payload["system"] = system_blocks

        if tools:
            payload["tools"] = tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif isinstance(tool_choice, dict):
                func_name = tool_choice.get("function", {}).get("name", "")
                if func_name:
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": _anthropic_tool_name(func_name),
                    }

        reasoning_effort = kwargs.get("reasoning_effort")

        # Opus models always use thinking (matching antigravity provider behavior)
        is_opus = "opus" in model.lower()
        if is_opus and not reasoning_effort:
            reasoning_effort = "medium"

        reasoning_effort_str = (
            str(reasoning_effort).lower().strip() if reasoning_effort else ""
        )
        reasoning_disabled = reasoning_effort is not None and reasoning_effort_str in (
            "none",
            "disabled",
            "off",
            "false",
            "disable",
        )

        if self._model_supports_adaptive_thinking(model):
            if not reasoning_disabled:
                payload["thinking"] = {"type": "adaptive"}

                if reasoning_effort_str in ("low", "medium", "high", "max"):
                    payload["output_config"] = {"effort": reasoning_effort_str}
        elif reasoning_effort and not reasoning_disabled:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._reasoning_effort_to_budget(
                    reasoning_effort, kwargs.get("max_tokens", 16384)
                ),
            }

        # Context management: preserve thinking blocks across turns
        if payload.get("thinking") and payload["thinking"].get("type") != "disabled":
            payload["context_management"] = {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": "all",
                    }
                ]
            }

        speed = kwargs.get("speed")
        if speed:
            payload["speed"] = speed

        # Only send temperature when thinking is disabled — API requires
        # temperature=1 when thinking is enabled, which is already the default.
        has_thinking = (
            "thinking" in payload and payload["thinking"].get("type") != "disabled"
        )
        if kwargs.get("temperature") is not None and not has_thinking:
            payload["temperature"] = kwargs["temperature"]

        payload = self._prefix_tool_names(payload)
        payload = self._inject_cache_control(payload)
        return payload

    def _reasoning_effort_to_budget(self, effort: Any, max_tokens: int) -> int:
        effort_str = str(effort).lower().strip()
        budget_map = {
            "low": 4096,
            "medium": 8192,
            "high": 16384,
        }
        budget = budget_map.get(effort_str)
        if budget is not None:
            return min(budget, max(max_tokens - 1000, 4096))
        try:
            return int(effort_str)
        except (ValueError, TypeError):
            return min(8192, max(max_tokens - 1000, 4096))

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs.get("model", "")
        file_logger = ProviderLogger(transaction_context)

        async def make_request():
            access_token = await self.get_access_token(credential_path)
            payload = self._build_anthropic_payload(kwargs)
            headers = self._build_anthropic_headers(
                access_token,
                payload,
                transaction_context,
            )

            file_logger.log_request(payload)

            url = f"{ANTHROPIC_API_BASE}/v1/messages?beta=true"
            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            )

        async def stream_handler(response_stream, attempt=1):
            stream_state: Dict[str, Any] = {}
            try:
                async with response_stream as response:
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_text = (
                            error_text.decode("utf-8")
                            if isinstance(error_text, bytes)
                            else error_text
                        )

                        if response.status_code == 401 and attempt == 1:
                            lib_logger.warning(
                                "Anthropic returned 401. Forcing token refresh and retrying."
                            )
                            await self._refresh_token(credential_path, force=True)
                            retry_stream = await make_request()
                            async for chunk in stream_handler(retry_stream, attempt=2):
                                yield chunk
                            return

                        if response.status_code == 429:
                            raise RateLimitError(
                                f"Anthropic rate limit: {error_text}",
                                llm_provider="anthropic",
                                model=model,
                                response=response,
                            )

                        error_msg = (
                            f"Anthropic HTTP {response.status_code}: {error_text}"
                        )
                        file_logger.log_error(error_msg)
                        if response.status_code == 400:
                            # Dump full payload to file for debugging opaque 400s
                            payload = self._build_anthropic_payload(kwargs)
                            msg_summary = []
                            for m in payload.get("messages", []):
                                role = m.get("role", "?")
                                c = m.get("content", "")
                                clen = (
                                    len(json.dumps(c))
                                    if not isinstance(c, str)
                                    else len(c)
                                )
                                msg_summary.append(f"{role}({clen})")
                            lib_logger.warning(
                                f"Anthropic 400 debug: model={payload.get('model')}, "
                                f"msgs=[{', '.join(msg_summary)}], "
                                f"has_tools={bool(payload.get('tools'))}, "
                                f"has_thinking={bool(payload.get('thinking'))}"
                            )
                            # Write full payload for inspection
                            try:
                                dump_path = Path("/tmp/anthropic_400_payload.json")
                                dump_path.write_text(
                                    json.dumps(payload, indent=2, default=str)
                                )
                                lib_logger.warning(
                                    f"Anthropic 400 payload dumped to {dump_path}"
                                )
                            except Exception as dump_err:
                                lib_logger.warning(
                                    f"Failed to dump 400 payload: {dump_err}"
                                )
                        raise httpx.HTTPStatusError(
                            error_msg,
                            request=response.request,
                            response=response,
                        )

                    current_event = None
                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)

                        if line.startswith("event:"):
                            current_event = line[6:].strip()
                            continue

                        if line.startswith("data:"):
                            data_str = (
                                line[5:].strip()
                                if line.startswith("data: ")
                                else line[5:]
                            )
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if current_event:
                                for chunk in self._anthropic_event_to_openai_chunks(
                                    current_event, data, model, stream_state
                                ):
                                    yield litellm.ModelResponse(**chunk)

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                file_logger.log_error(f"Error during Anthropic stream: {e}")
                lib_logger.error(f"Anthropic stream error: {e}", exc_info=True)
                raise

        async def logging_stream_wrapper():
            chunks = []
            try:
                async for chunk in stream_handler(await make_request()):
                    chunks.append(chunk)
                    yield chunk
            finally:
                if chunks:
                    final = self._stream_to_completion_response(chunks)
                    file_logger.log_final_response(final.dict())

        if kwargs.get("stream"):
            return logging_stream_wrapper()
        else:

            async def non_stream():
                all_chunks = [c async for c in logging_stream_wrapper()]
                return self._stream_to_completion_response(all_chunks)

            return await non_stream()

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        if not chunks:
            raise ValueError("No chunks to reassemble")

        final_message = {"role": "assistant"}
        aggregated_tool_calls = {}
        usage_data = None
        finish_reason = "stop"
        first_chunk = chunks[0]

        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            if hasattr(choice, "get"):
                delta = choice.get("delta", {})
                choice_finish = choice.get("finish_reason")
            elif hasattr(choice, "delta"):
                delta = choice.delta if choice.delta else {}
                if hasattr(delta, "model_dump"):
                    delta = delta.model_dump(exclude_none=True)
                elif hasattr(delta, "__dict__") and not isinstance(delta, dict):
                    delta = {
                        k: v
                        for k, v in delta.__dict__.items()
                        if not k.startswith("_") and v is not None
                    }
                choice_finish = getattr(choice, "finish_reason", None)
            else:
                delta = {}
                choice_finish = None

            if delta.get("content"):
                final_message.setdefault("content", "")
                final_message["content"] += delta["content"]

            if delta.get("reasoning_content"):
                final_message.setdefault("reasoning_content", "")
                final_message["reasoning_content"] += delta["reasoning_content"]

            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                idx = tc.get("index", 0)
                if idx not in aggregated_tool_calls:
                    aggregated_tool_calls[idx] = {
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    aggregated_tool_calls[idx]["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    aggregated_tool_calls[idx]["function"]["name"] += func["name"]
                if func.get("arguments"):
                    aggregated_tool_calls[idx]["function"]["arguments"] += func[
                        "arguments"
                    ]

            if choice_finish:
                finish_reason = choice_finish

        for chunk in reversed(chunks):
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
                break

        if aggregated_tool_calls:
            final_message["tool_calls"] = list(aggregated_tool_calls.values())
            finish_reason = "tool_calls"

        for f in ["content", "tool_calls", "function_call"]:
            if f not in final_message:
                final_message[f] = None

        return litellm.ModelResponse(
            **{
                "id": first_chunk.id,
                "object": "chat.completion",
                "created": first_chunk.created,
                "model": first_chunk.model,
                "choices": [
                    {
                        "index": 0,
                        "message": final_message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage_data,
            }
        )
