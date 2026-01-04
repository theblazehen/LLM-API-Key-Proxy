import json
import os
import csv
from datetime import datetime
from pathlib import Path
import uuid
from typing import Literal, Dict, Optional, Any
import logging

from .provider_urls import get_provider_endpoint

# Configure standard logger
logger = logging.getLogger(__name__)


def log_request_to_console(
    url: str, headers: dict, client_info: tuple, request_data: dict
):
    """
    Logs a concise, single-line summary of an incoming request to the console.
    """
    time_str = datetime.now().strftime("%H:%M")
    model_full = request_data.get("model", "N/A")

    provider = "N/A"
    model_name = model_full
    endpoint_url = "N/A"

    if "/" in model_full:
        parts = model_full.split("/", 1)
        provider = parts[0]
        model_name = parts[1]
        # Use the helper function to get the full endpoint URL
        endpoint_url = get_provider_endpoint(provider, model_name, url) or "N/A"

    log_message = f"{time_str} - {client_info[0]}:{client_info[1]} - provider: {provider}, model: {model_name} - {endpoint_url}"
    logging.info(log_message)


class CSVLogger:
    """
    Logs request details to a CSV file for analysis.
    Rotates files daily to avoid massive files.
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_file_ready()

    def _get_log_file_path(self) -> Path:
        """Get the main log file path."""
        return self.logs_dir / "usage_log.csv"

    def _ensure_file_ready(self):
        """Ensure the current log file exists and has headers."""
        file_path = self._get_log_file_path()
        if not file_path.exists():
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "request_id",
                        "model",
                        "provider",
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_tokens",
                        "total_tokens",
                        "cost_usd",
                        "duration_seconds",
                        "status",
                    ]
                )

    def log_completion(
        self,
        model: str,
        usage: Dict[str, Any],
        duration: float,
        cost: float = 0.0,
        status: str = "success",
        request_id: str = "",
    ):
        """
        Log a completed request to the CSV file.

        Args:
            model: Full model name (e.g. "antigravity/claude-3-opus")
            usage: Usage dictionary from response (prompt_tokens, completion_tokens, etc.)
            duration: Request duration in seconds
            cost: Estimated cost in USD
            status: Request status (success/error)
            request_id: Unique request ID
        """
        try:
            self._ensure_file_ready()

            # Extract provider if present
            provider = "unknown"
            if "/" in model:
                provider = model.split("/")[0]

            # Extract token counts
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Extract cached tokens (support multiple formats)
            cached_tokens = 0
            if "prompt_tokens_details" in usage:
                details = usage["prompt_tokens_details"]
                if isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens", 0)
                elif hasattr(details, "cached_tokens"):
                    cached_tokens = details.cached_tokens
            elif "cache_read_input_tokens" in usage:
                cached_tokens = usage["cache_read_input_tokens"]
            elif "cached_tokens" in usage:  # Our normalized field
                cached_tokens = usage["cached_tokens"]

            timestamp = datetime.now().isoformat()

            with open(
                self._get_log_file_path(), "a", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        timestamp,
                        request_id,
                        model,
                        provider,
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        total_tokens,
                        f"{cost:.6f}",
                        f"{duration:.2f}",
                        status,
                    ]
                )

        except Exception as e:
            logger.error(f"Failed to write to CSV log: {e}")
