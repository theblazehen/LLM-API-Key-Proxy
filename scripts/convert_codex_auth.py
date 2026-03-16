#!/usr/bin/env python3
"""
One-off conversion: ~/.codex/auth.json -> oauth_creds/codex_oauth_1.json

Converts the native Codex CLI auth format to the proxy's flat credential format.
"""

import base64
import json
import sys
import time
from pathlib import Path


def decode_jwt_payload(token: str) -> dict:
    payload = token.split(".")[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    return json.loads(base64.urlsafe_b64decode(payload))


def convert(src: Path, dst: Path):
    with open(src) as f:
        raw = json.load(f)

    tokens = raw.get("tokens", {})
    access_token = tokens.get("access_token")
    if not access_token:
        print(f"No access_token found in {src}", file=sys.stderr)
        sys.exit(1)

    claims = decode_jwt_payload(access_token)

    auth_info = claims.get("https://api.openai.com/auth", {})
    profile = claims.get("https://api.openai.com/profile", {})

    result = {
        "access_token": access_token,
        "refresh_token": tokens.get("refresh_token"),
        "id_token": tokens.get("id_token"),
        "account_id": tokens.get("account_id") or auth_info.get("chatgpt_account_id"),
        "expiry_date": claims.get("exp", 0),
        "_proxy_metadata": {
            "email": profile.get("email", "unknown"),
            "plan_type": auth_info.get("chatgpt_plan_type"),
            "last_check_timestamp": time.time(),
        },
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(result, f, indent=2)
    dst.chmod(0o600)

    print(f"Converted {src} -> {dst}")
    print(f"  email:      {result['_proxy_metadata']['email']}")
    print(f"  plan:       {result['_proxy_metadata']['plan_type']}")
    print(f"  account_id: {result['account_id']}")
    print(f"  expires:    {time.ctime(result['expiry_date'])}")


if __name__ == "__main__":
    src = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".codex" / "auth.json"
    )
    dst = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("oauth_creds/codex_oauth_1.json")
    )

    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    if dst.exists():
        print(f"Destination already exists: {dst}")
        print("Overwrite? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            sys.exit(0)

    convert(src, dst)
