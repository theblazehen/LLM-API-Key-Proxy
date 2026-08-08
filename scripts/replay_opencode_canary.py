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


def keep_first_n_chars_in_system(payload: dict, count: int) -> None:
    keep_system_slice(payload, 0, count)


def keep_system_slice(payload: dict, start: int, length: int) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    start = max(start, 0)
    remaining_start = start
    remaining_length = max(length, 0)
    new_messages = []

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            new_messages.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, str):
            new_messages.append(msg)
            continue

        if remaining_start >= len(content):
            remaining_start -= len(content)
            continue

        slice_start = remaining_start
        slice_end = min(len(content), slice_start + remaining_length)
        sliced = content[slice_start:slice_end]
        remaining_length = max(remaining_length - len(sliced), 0)
        remaining_start = 0

        if sliced:
            copied = deepcopy(msg)
            copied["content"] = sliced
            new_messages.append(copied)

    payload["messages"] = new_messages


def transform_system_text(payload: dict, mode: str) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    def rot13_char(ch: str) -> str:
        if "a" <= ch <= "z":
            return chr((ord(ch) - ord("a") + 13) % 26 + ord("a"))
        if "A" <= ch <= "Z":
            return chr((ord(ch) - ord("A") + 13) % 26 + ord("A"))
        return ch

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if mode == "rot13":
            msg["content"] = "".join(rot13_char(ch) for ch in content)
        elif mode == "filler":
            msg["content"] = "A" * len(content)
        elif mode == "xmask":
            msg["content"] = "".join("X" if not ch.isspace() else ch for ch in content)
        elif mode == "html":
            msg["content"] = (
                content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
        elif mode == "env-prose":
            content = content.replace(
                "Here is some useful information about the environment you are running in:\n<env>\n  Working directory: /home/jasmin\n  Workspace root folder: /\n  Is directory a git repo: no\n  Platform: linux\n  Today's date: Tue Apr 14 2026\n</env>\n<directories>\n  \n</directories>",
                "Environment summary: Linux machine, not currently in a git repo, working from /home/jasmin with workspace root at /. Date: Tue Apr 14 2026.",
            )
            content = content.replace("<env>", "Environment:\n")
            content = content.replace("</env>", "")
            content = content.replace(
                "<directories>\n  \n</directories>", "Directories: none listed."
            )
            msg["content"] = content


def replace_in_system(payload: dict, needle: str, replacement: str) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not needle:
        return
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = content.replace(needle, replacement)


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


def drop_reasoning(payload: dict) -> None:
    payload.pop("reasoning_effort", None)


def set_model(payload: dict, model: str) -> None:
    payload["model"] = model


def apply_variant(
    payload: dict,
    variant: str,
    model: str | None,
    first_n_tools: int | None,
    tool_names: list[str] | None,
    system_chars: int | None,
    system_slice_start: int | None,
    system_slice_length: int | None,
    system_transform: str | None,
    system_replace: list[tuple[str, str]] | None,
) -> dict:
    mutated = deepcopy(payload)

    if model:
        set_model(mutated, model)

    if first_n_tools is not None:
        keep_first_n_tools(mutated, first_n_tools)

    if tool_names is not None:
        keep_named_tools(mutated, tool_names)

    if system_chars is not None:
        keep_first_n_chars_in_system(mutated, system_chars)

    if system_slice_start is not None and system_slice_length is not None:
        keep_system_slice(mutated, system_slice_start, system_slice_length)

    if system_transform is not None:
        transform_system_text(mutated, system_transform)

    if system_replace is not None:
        for needle, replacement in system_replace:
            replace_in_system(mutated, needle, replacement)

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
    if variant == "drop-reasoning":
        drop_reasoning(mutated)
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
            "drop-reasoning",
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
        "--system-chars",
        type=int,
        default=None,
        help="Keep only the first N characters across system messages",
    )
    parser.add_argument(
        "--system-slice-start",
        type=int,
        default=None,
        help="Start offset for a contiguous system prompt slice",
    )
    parser.add_argument(
        "--system-slice-length",
        type=int,
        default=None,
        help="Length for a contiguous system prompt slice",
    )
    parser.add_argument(
        "--system-transform",
        choices=["rot13", "filler", "xmask", "html", "env-prose"],
        default=None,
        help="Transform system prompt text while preserving overall size",
    )
    parser.add_argument(
        "--system-replace",
        action="append",
        default=None,
        help="Replace system text in the form needle=replacement",
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
    system_replace = None
    if args.system_replace:
        system_replace = []
        for item in args.system_replace:
            needle, sep, replacement = item.partition("=")
            if not sep:
                raise ValueError(
                    "--system-replace must be in the form needle=replacement"
                )
            system_replace.append((needle, replacement))

    replay_payload = apply_variant(
        captured,
        args.variant,
        args.model,
        args.first_n_tools,
        tool_names,
        args.system_chars,
        args.system_slice_start,
        args.system_slice_length,
        args.system_transform,
        system_replace,
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
