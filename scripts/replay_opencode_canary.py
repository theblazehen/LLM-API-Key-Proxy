#!/usr/bin/env python3
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import httpx


DEFAULT_URL = "https://llmapi.llm.blazelight.dev/v1/chat/completions"
DEFAULT_API_KEY = "sk-hunter2"
CANARY_TEXT = "You're out of extra usage"


def load_capture(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    body = data.get("body_json")
    if not isinstance(body, dict):
        raise ValueError(f"capture at {path} does not contain body_json")
    return body


def minify_user_message(payload: dict) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    payload["messages"] = [
        msg
        for msg in messages
        if not (isinstance(msg, dict) and msg.get("role") == "system")
    ]
    if not payload["messages"]:
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["messages"][-1] = {"role": "user", "content": "hello"}


def strip_tools(payload: dict) -> None:
    payload.pop("tools", None)
    payload.pop("tool_choice", None)


def keep_first_n_tools(payload: dict, count: int) -> None:
    tools = payload.get("tools")
    if isinstance(tools, list):
        payload["tools"] = tools[:count]
        if count == 0:
            payload.pop("tool_choice", None)


def keep_named_tools(payload: dict, names: list[str]) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    wanted = set(names)
    filtered = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if name in wanted:
            filtered.append(tool)
    payload["tools"] = filtered
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        chosen = tool_choice.get("function", {}).get("name")
        if chosen not in wanted:
            payload.pop("tool_choice", None)


def titlecase_tool_names(payload: dict) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    name_map = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not isinstance(name, str):
            continue
        new_name = "".join(part.capitalize() for part in name.split("_"))
        name_map[name] = new_name
        func["name"] = new_name
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function")
        if isinstance(func, dict):
            name = func.get("name")
            if name in name_map:
                func["name"] = name_map[name]


def strip_tool_descriptions(payload: dict) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if isinstance(func, dict):
            func["description"] = ""


def minimize_tool_schemas(payload: dict) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if isinstance(func, dict):
            func["parameters"] = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }


def strip_stream_options(payload: dict) -> None:
    payload.pop("stream_options", None)


def disable_reasoning(payload: dict) -> None:
    payload["reasoning_effort"] = "none"


def set_model(payload: dict, model: str) -> None:
    payload["model"] = model


def apply_variant(
    payload: dict,
    variant: str,
    model: str | None,
    first_n_tools: int | None,
    tool_names: list[str] | None,
) -> dict:
    mutated = deepcopy(payload)

    if model:
        set_model(mutated, model)

    if first_n_tools is not None:
        keep_first_n_tools(mutated, first_n_tools)

    if tool_names is not None:
        keep_named_tools(mutated, tool_names)

    if variant == "captured":
        return mutated
    if variant == "no-system":
        minify_user_message(mutated)
        return mutated
    if variant == "no-tools":
        strip_tools(mutated)
        return mutated
    if variant == "no-system-no-tools":
        minify_user_message(mutated)
        strip_tools(mutated)
        return mutated
    if variant == "no-stream-options":
        strip_stream_options(mutated)
        return mutated
    if variant == "no-reasoning":
        disable_reasoning(mutated)
        return mutated
    if variant == "titlecase-tools":
        titlecase_tool_names(mutated)
        return mutated
    if variant == "no-tool-descriptions":
        strip_tool_descriptions(mutated)
        return mutated
    if variant == "minimal-tool-schemas":
        minimize_tool_schemas(mutated)
        return mutated
    if variant == "minimal":
        minify_user_message(mutated)
        strip_tools(mutated)
        strip_stream_options(mutated)
        disable_reasoning(mutated)
        mutated["max_tokens"] = 256
        return mutated

    raise ValueError(f"unknown variant: {variant}")


def send(url: str, api_key: str, payload: dict) -> tuple[int, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, json=payload)
    return response.status_code, response.text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay captured OpenCode request variants"
    )
    parser.add_argument("capture", type=Path, help="Path to captured request JSON")
    parser.add_argument(
        "--variant",
        default="captured",
        choices=[
            "captured",
            "no-system",
            "no-tools",
            "no-system-no-tools",
            "no-stream-options",
            "no-reasoning",
            "titlecase-tools",
            "no-tool-descriptions",
            "minimal-tool-schemas",
            "minimal",
        ],
        help="Payload mutation to apply before replay",
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--first-n-tools",
        type=int,
        default=None,
        help="Keep only the first N tools from the captured payload",
    )
    parser.add_argument(
        "--tool-names",
        default=None,
        help="Comma-separated list of tool names to keep from the captured payload",
    )
    parser.add_argument(
        "--dump-payload",
        action="store_true",
        help="Print the replay payload before sending",
    )
    args = parser.parse_args()

    captured = load_capture(args.capture)
    tool_names = None
    if args.tool_names:
        tool_names = [
            name.strip() for name in args.tool_names.split(",") if name.strip()
        ]
    replay_payload = apply_variant(
        captured,
        args.variant,
        args.model,
        args.first_n_tools,
        tool_names,
    )

    if args.dump_payload:
        print(json.dumps(replay_payload, indent=2, ensure_ascii=False))
        return 0

    status_code, text = send(args.url, args.api_key, replay_payload)
    print(f"status={status_code}")
    print(f"canary={'yes' if CANARY_TEXT in text else 'no'}")
    print(text)
    return 0 if status_code < 500 else 1


if __name__ == "__main__":
    sys.exit(main())
