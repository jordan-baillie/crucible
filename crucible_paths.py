"""Central path/config resolution for Crucible — every external coupling in ONE place.

All paths are env-overridable so the system runs on any box:
  CRUCIBLE_ROOT     repo root            (default: this file's directory)
  CRUCIBLE_WIKI     research wiki repo   (shared state: queue/locks/elite/FDR registry/pages)
  CRUCIBLE_DATA     market-data root     (Sharadar/cache layout, see sdk/adapters.py)
  CRUCIBLE_SECRETS  secrets JSON         (telegram_bot_token, telegram_chat_id, fred_api_key)
  CRUCIBLE_DEPLOY   paper-deploy target  (Atlas-style live contract dir; empty disables deploy)
  MODEL_POLICY      central model policy (tiers: frontier/standard/fast)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("CRUCIBLE_ROOT", Path(__file__).resolve().parent))
WIKI = Path(os.environ.get("CRUCIBLE_WIKI", "/root/research-wiki"))
DATA = Path(os.environ.get("CRUCIBLE_DATA", "/root/atlas/data"))
SECRETS = Path(os.environ.get("CRUCIBLE_SECRETS", os.path.expanduser("~/.atlas-secrets.json")))
# Paper-deploy target: a directory implementing the deploy contract (see live/deploy.py docstring).
# Set CRUCIBLE_DEPLOY="" (empty) to disable deployment entirely — verdicts still record normally.
_deploy = os.environ.get("CRUCIBLE_DEPLOY", "/root/atlas")
DEPLOY_TARGET = Path(_deploy) if _deploy else None
MODEL_POLICY = Path(os.environ.get("MODEL_POLICY", "/root/.pi/model-policy.json"))

STRATEGIES = ROOT / "strategies"
LOGS = ROOT / "logs"
RUN_LOG = ROOT / "agent" / "run_log.jsonl"
KILLSWITCH = ROOT / "LOOP_DISABLED"
FORGE_MODE_FILE = ROOT / "FORGE_MODE"


def forge_mode() -> str:
    """Forge cadence mode — the SINGLE source of truth, read by the forge/drip systemd
    ExecCondition gates so one flag switches cadence with no double FDR-family spend:
      'batch' (default) nightly 3-smith burst at 03:30 (crucible-forge);
      'drip'            1 smith every 3h (crucible-drip).
    Absent/garbage file -> 'batch' (fail safe to the proven baseline). Flip with
    `echo drip > FORGE_MODE` (and back). Presence-tolerant, never raises.
    """
    try:
        m = FORGE_MODE_FILE.read_text(encoding="utf-8").strip().lower()
    except Exception:
        m = ""
    return m if m in ("batch", "drip") else "batch"
QUEUE = Path(os.environ.get("CRUCIBLE_QUEUE", os.environ.get("HEPH_QUEUE", WIKI / ".queue" / "queue.jsonl")))  # HEPH_QUEUE legacy compat
LOCKS = WIKI / ".locks"
ELITE = WIKI / ".elite" / "pool.jsonl"
REGISTRY = WIKI / ".registry"
