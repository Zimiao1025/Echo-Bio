from __future__ import annotations

import json

import pytest

from echo_bio import cli
from echo_bio.user_runtime import UserRunConfig, _validate_data


def test_provider_check_dry_run_needs_no_optional_dependency(capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert cli.main.__module__ == "echo_bio.cli"
    args = cli.parser().parse_args(["provider-check"])
    assert args.handler(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run"
    assert report["profile"]["api_key_present"] is False
    assert report["profile"]["automatic_retry"] is False


def test_run_requires_both_explicit_risk_confirmations():
    args = cli.parser().parse_args(["run", "--task", "test"])
    with pytest.raises(SystemExit, match="--confirm-paid"):
        args.handler(args)


def test_data_validation_is_local_and_content_free(tmp_path):
    data = tmp_path / "matrix.csv"
    data.write_text("gene,value\nA,1\n", encoding="utf-8")
    validated = _validate_data(((data, "small expression matrix"),))
    assert validated == ((data.resolve(), "small expression matrix"),)


def test_user_config_has_conservative_request_default(tmp_path):
    config = UserRunConfig(task="summarize", workspace=tmp_path)
    assert config.max_requests == 8
    assert config.timeout_seconds == 600
