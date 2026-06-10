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
DEPLOY_TARGET = Path(os.environ.get("CRUCIBLE_DEPLOY", "/root/atlas"))
MODEL_POLICY = Path(os.environ.get("MODEL_POLICY", "/root/.pi/model-policy.json"))

STRATEGIES = ROOT / "strategies"
LOGS = ROOT / "logs"
RUN_LOG = ROOT / "agent" / "run_log.jsonl"
KILLSWITCH = ROOT / "LOOP_DISABLED"
QUEUE = Path(os.environ.get("HEPH_QUEUE", WIKI / ".queue" / "queue.jsonl"))  # HEPH_QUEUE kept for compat
LOCKS = WIKI / ".locks"
ELITE = WIKI / ".elite" / "pool.jsonl"
REGISTRY = WIKI / ".registry"
