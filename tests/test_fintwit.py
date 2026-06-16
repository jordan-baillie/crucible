"""Credibility-vetted FinTwit idea-grounding layer (agent.fintwit). Mirrors the Firecrawl layer:
additive + GRACEFUL — any artifact problem returns empty so the scout runs exactly as today."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from agent import fintwit as F


def _digest(generated_at=None, accounts=None, schema_version=1):
    return {
        "schema_version": schema_version,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "n_accounts": len(accounts or []),
        "accounts": accounts if accounts is not None else [
            {"handle": "GoodCaller", "primary_assets": "BTC,ETH",
             "credibility": {"recency_weighted_hit_rate": 0.66, "resolvable_claims": 18},
             "recent_posts": [{"posted_at": "2026-06-15", "text": "Negative funding means shorts are paying — crowded-short squeeze risk building."}]},
        ],
    }


@pytest.fixture()
def digest_at(tmp_path, monkeypatch):
    p = tmp_path / "fintwit_digest.json"
    monkeypatch.setenv("CRUCIBLE_FINTWIT_DIGEST", str(p))
    return p


def test_absent_digest_is_graceful(digest_at):
    accounts, err = F.read_digest()
    assert accounts == [] and "no fintwit digest" in err
    assert "unavailable" in F.fintwit_text()           # never raises, returns a reason


def test_unparseable_digest_is_graceful(digest_at):
    digest_at.write_text("{not json")
    accounts, err = F.read_digest()
    assert accounts == [] and "unparseable" in err


def test_wrong_schema_version_rejected(digest_at):
    digest_at.write_text(json.dumps(_digest(schema_version=999)))
    accounts, err = F.read_digest()
    assert accounts == [] and "schema_version" in err


def test_stale_digest_ignored(digest_at):
    old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    digest_at.write_text(json.dumps(_digest(generated_at=old)))
    accounts, err = F.read_digest()
    assert accounts == [] and "stale" in err


def test_valid_digest_reads_and_formats(digest_at):
    digest_at.write_text(json.dumps(_digest()))
    accounts, err = F.read_digest()
    assert err == "" and len(accounts) == 1
    txt = F.format_fintwit(accounts)
    assert "@GoodCaller" in txt
    assert "recency-weighted hit-rate 0.66" in txt
    assert "squeeze risk" in txt                        # the reasoning post is included


def test_fintwit_text_uses_env_override(digest_at):
    digest_at.write_text(json.dumps(_digest()))
    assert "@GoodCaller" in F.fintwit_text()
