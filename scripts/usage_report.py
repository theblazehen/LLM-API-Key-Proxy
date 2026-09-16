#!/usr/bin/env python3
"""
Usage Report Script for LLM API Key Proxy.
Calculates the estimated "free" value of daily usage based on token counts.
"""

import json
import os
import csv
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from collections import defaultdict

# Default Pricing per 1M tokens (USD)
# Used as fallback if proxy API is unreachable
PRICING = {
    # Gemini 2.5 (Flash pricing) - Cached is ~25% of input
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30, "cached": 0.01875},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30, "cached": 0.01875},
    # Gemini 2.5 Pro (Pro pricing) - Cached is ~25% of input
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00, "cached": 0.3125},
    # Gemini 3 (Assumed Pro pricing)
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.00, "cached": 0.3125},
    "gemini-3-pro-image-preview": {"input": 2.50, "output": 10.00, "cached": 0.625},
    # Claude (Sonnet 3.5 pricing) - Cached is 10% of input
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "cached": 0.30},
    # Claude Opus (Opus 3 pricing) - Cached is 10% of input
    "claude-opus-4-5": {"input": 15.00, "output": 75.00, "cached": 1.50},
}


def fetch_live_pricing(base_url: str = "http://127.0.0.1:8000"):
    """Fetch accurate pricing from running proxy instance."""
    try:
        # Use a short timeout so we don't hang if proxy is down
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=0.5) as response:
            if response.status != 200:
                return

            data = json.load(response)
            models = data.get("data", [])

            count = 0
            for model in models:
                model_id = model.get("id")
                pricing = model.get("pricing")

                if model_id and pricing:
                    # Clean the ID to match our keys (remove provider prefix if needed, though PRICING keys here are mixed)
                    # The script's calculate_cost strips prefixes, so we should key PRICING by the bare model name
                    # OR we update PRICING to support full IDs.
                    # Currently calculate_cost does: clean_name = model_name.split("/")[-1]
                    # So we should store by short name.

                    short_name = (
                        model_id.split("/")[-1] if "/" in model_id else model_id
                    )

                    # Convert pricing to per-1M-tokens
                    # API returns cost PER TOKEN (e.g. 1.25e-06)
                    # We convert to per 1M (multiply by 1e6)
                    PRICING[short_name] = {
                        "input": pricing.get("prompt", 0) * 1_000_000,
                        "output": pricing.get("completion", 0) * 1_000_000,
                        "cached": pricing.get("cached_input", 0) * 1_000_000,
                    }
                    count += 1

            if count > 0:
                print(f"✓ Synced pricing for {count} models from local proxy.")

    except (urllib.error.URLError, ConnectionRefusedError, Exception):
        # Proxy likely not running, use defaults
        print("! Proxy unreachable, using default pricing table.")


def get_key_usage_path() -> Path:
    """Find key_usage.json in current or parent directory."""
    current = Path.cwd()
    candidates = [
        current / "key_usage.json",
        current.parent / "key_usage.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return current / "key_usage.json"


def calculate_cost(
    model_name: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> float:
    """Calculate cost based on pricing table."""
    # Strip provider prefix if present
    clean_name = model_name.split("/")[-1] if "/" in model_name else model_name
    # Strip thinking suffix
    clean_name = clean_name.replace(":thinking", "")

    pricing = PRICING.get(clean_name)
    if not pricing:
        # Fallback pricing (average of flash/pro)
        if "flash" in clean_name:
            pricing = PRICING["gemini-2.5-flash"]
        elif "claude" in clean_name:
            pricing = PRICING["claude-sonnet-4-5"]
        else:
            pricing = PRICING["gemini-2.5-pro"]

    # Calculate costs
    # Note: cached_tokens are usually included in total input_tokens by providers
    # So we subtract them to get "fresh" input tokens
    fresh_input = max(0, input_tokens - cached_tokens)

    input_cost = (fresh_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_tokens / 1_000_000) * pricing.get(
        "cached", pricing["input"] * 0.25
    )
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + cached_cost + output_cost


def main():
    # Try to sync pricing first
    fetch_live_pricing()

    usage_path = get_key_usage_path()
    if not usage_path.exists():
        print(f"Error: Could not find {usage_path}")
        return

    try:
        with open(usage_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading usage file: {e}")
        return

    print(f"\n{'=' * 85}")
    print(f" LLM Usage Report - Estimated 'Free' Savings")
    print(f"{'=' * 85}")
    print(f"Source: {usage_path}")
    print(f"Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'-' * 85}")

    total_daily_cost = 0.0
    total_daily_input = 0
    total_daily_cached = 0
    total_daily_output = 0

    # Aggregated stats per model
    model_stats = {}

    for key, key_data in data.items():
        daily = key_data.get("daily", {})
        models = daily.get("models", {})

        # Merge per-model structure if present
        if "models" in key_data and isinstance(key_data["models"], dict):
            # Check if this is the new per-model structure (has window_start_ts etc)
            # or just a merge target. The UsageManager now puts everything in key_data["models"]
            # if reset_mode is per_model.
            # But the script needs to handle both locations to be safe or just use the one that has data.
            # Actually UsageManager puts data in EITHER daily OR models depending on mode.
            # Let's check both.
            pass

        # Smart merge of daily and per-model stats
        sources = [models]
        if "models" in key_data and key_data["models"]:
            sources.append(key_data["models"])

        for source in sources:
            for model_name, stats in source.items():
                prompt_tokens = stats.get("prompt_tokens", 0)
                completion_tokens = stats.get("completion_tokens", 0)
                cached_tokens = stats.get("cached_tokens", 0)

                # If cached_tokens is 0 but we have a lot of prompt tokens,
                # we might want to estimate for legacy data, but let's stick to actuals for now
                # to encourage using the new system.

                cost = calculate_cost(
                    model_name, prompt_tokens, completion_tokens, cached_tokens
                )

                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "input": 0,
                        "cached": 0,
                        "output": 0,
                        "cost": 0.0,
                        "count": 0,
                    }

                # Avoid double counting if iterating multiple sources?
                # No, UsageManager uses EITHER daily OR models, not both for the same key at same time.
                # But to be safe against duplicates if keys switched modes, we just sum up.

                model_stats[model_name]["input"] += prompt_tokens
                model_stats[model_name]["cached"] += cached_tokens
                model_stats[model_name]["output"] += completion_tokens
                model_stats[model_name]["cost"] += cost
                model_stats[model_name]["count"] += stats.get("success_count", 0)

                total_daily_cost += cost
                total_daily_input += prompt_tokens
                total_daily_cached += cached_tokens
                total_daily_output += completion_tokens

    # Print Report
    print(
        f"\n{'Model':<35} {'Reqs':<6} {'Input':<10} {'Cached':<10} {'Output':<10} {'Est. Savings':<12}"
    )
    print(f"{'-' * 35} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

    for model, stats in sorted(
        model_stats.items(), key=lambda x: x[1]["cost"], reverse=True
    ):
        print(
            f"{model:<35} {stats['count']:<6} {stats['input']:<10,} {stats['cached']:<10,} {stats['output']:<10,} ${stats['cost']:.4f}"
        )

    print(f"{'-' * 85}")
    print(
        f"{'Current Window (Daily)':<35} {'':<6} {total_daily_input:<10,} {total_daily_cached:<10,} {total_daily_output:<10,} ${total_daily_cost:.4f}"
    )
    print(f"{'=' * 85}\n")

    # Global/Lifetime stats (Global archive + Current Daily)
    print(f"Lifetime Estimated Savings (Including Today):")
    total_lifetime_cost = 0.0

    # 1. Sum up global archived stats
    for key, key_data in data.items():
        glob = key_data.get("global", {})
        models = glob.get("models", {})
        for model_name, stats in models.items():
            cost = calculate_cost(
                model_name,
                stats.get("prompt_tokens", 0),
                stats.get("completion_tokens", 0),
                stats.get("cached_tokens", 0),
            )
            total_lifetime_cost += cost

    # 2. Add current daily stats (since they haven't been archived yet)
    total_lifetime_cost += total_daily_cost

    print(f"  ${total_lifetime_cost:,.2f}")
    print(f"{'=' * 60}")

    # Daily breakdown from CSV log
    print_daily_breakdown()


def get_csv_log_path() -> Path:
    """Find usage_log.csv in logs directory."""
    current = Path.cwd()
    candidates = [
        current / "logs" / "usage_log.csv",
        current.parent / "logs" / "usage_log.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return current / "logs" / "usage_log.csv"


def print_daily_breakdown(days: int = 30):
    """Print daily savings breakdown from CSV log."""
    csv_path = get_csv_log_path()
    if not csv_path.exists():
        print(f"\n! No CSV log found at {csv_path}")
        print("  (CSV logging starts after the first request with the updated proxy)")
        return

    daily_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"requests": 0, "input": 0, "cached": 0, "output": 0, "savings": 0.0}
    )

    cutoff = datetime.now() - timedelta(days=days)

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    if ts < cutoff:
                        continue

                    date_key = ts.strftime("%Y-%m-%d")
                    model = row.get("model", "unknown")
                    prompt = int(row.get("prompt_tokens", 0))
                    completion = int(row.get("completion_tokens", 0))
                    cached = int(row.get("cached_tokens", 0))

                    cost = calculate_cost(model, prompt, completion, cached)

                    daily_stats[date_key]["requests"] += 1
                    daily_stats[date_key]["input"] += prompt
                    daily_stats[date_key]["cached"] += cached
                    daily_stats[date_key]["output"] += completion
                    daily_stats[date_key]["savings"] += cost
                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print(f"\n! Error reading CSV log: {e}")
        return

    if not daily_stats:
        print(f"\n! No data in CSV log for the last {days} days")
        return

    print(f"\n{'=' * 75}")
    print(f" Daily Breakdown (Last {days} Days)")
    print(f"{'=' * 75}")
    print(
        f"{'Date':<12} {'Reqs':<8} {'Input':<12} {'Cached':<12} {'Output':<12} {'Savings':<10}"
    )
    print(f"{'-' * 12} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")

    total_savings = 0.0
    for date in sorted(daily_stats.keys(), reverse=True):
        stats = daily_stats[date]
        total_savings += stats["savings"]
        print(
            f"{date:<12} {stats['requests']:<8} {stats['input']:<12,} {stats['cached']:<12,} {stats['output']:<12,} ${stats['savings']:.2f}"
        )

    print(f"{'-' * 75}")
    print(f"{'TOTAL':<12} {'':<8} {'':<12} {'':<12} {'':<12} ${total_savings:.2f}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
