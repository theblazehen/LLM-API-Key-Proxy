import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.transaction_logger import TransactionLogger


def _metadata(tmp_path, response_data):
    logger = TransactionLogger(
        "codex", "gpt-5.6-sol", parent_dir=tmp_path, enabled=True
    )
    logger.log_response(response_data)
    return json.loads((tmp_path / "openai" / "metadata.json").read_text())


def test_metadata_keeps_prompt_cache_read_and_write_counters(tmp_path):
    metadata = _metadata(
        tmp_path,
        {
            "model": "gpt-5.6-sol",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 20,
                "total_tokens": 140,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_tokens": 16,
                },
            },
        },
    )

    assert metadata["usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 20,
        "total_tokens": 140,
        "cached_tokens": 80,
        "cache_creation_tokens": 16,
    }


def test_metadata_omits_cache_counters_without_prompt_token_details(tmp_path):
    metadata = _metadata(
        tmp_path,
        {
            "model": "gpt-5.6-sol",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 20,
                "total_tokens": 140,
            },
        },
    )

    assert metadata["usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 20,
        "total_tokens": 140,
    }
