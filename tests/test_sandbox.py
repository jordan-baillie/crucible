"""Sandbox denylist table tests: code that MUST be rejected vs MUST be accepted."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.sandbox import scan_code  # noqa: E402

MUST_REJECT = [
    "import os\nos.system('rm -rf /')",
    "import subprocess",
    "from urllib.request import urlopen",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "df.to_csv('/tmp/leak.csv')",
    "df.to_parquet('out.pq')",
    "df.to_pickle('x.pkl')",
    "pd.read_pickle('x.pkl')",
    "np.save('arr.npy', a)",
    "pd.read_csv('https://evil.example.com/data.csv')",
    "pd.read_html('http://example.com')",
    "Path('x').write_text('data')",
    "__builtins__['open']('x')",
    # reflection-escape class (closed structurally, not per-instance): each of these reaches a banned
    # module/builtin WITHOUT naming it, and rlimits don't bound sockets/processes -> must be refused.
    "().__class__.__bases__[0].__subclasses__()",   # classic sandbox break to object subclasses
    "x.__class__.__mro__",
    "f.__globals__['os']",                          # function globals -> already-imported os
    "f.__code__.co_consts",
    "().__class__.__dict__",
    "getattr(obj, '__globals__')",                  # dunder via getattr
    "getattr(obj, '__' + 'globals__')",             # computed/concatenated name
    "getattr(obj, name)",                           # non-literal name (dynamic indirection)
    "getattr(obj, attr_name='__globals__')",        # kwarg name -> no positional literal
    "import sys\nsys.modules['os'].system('id')",   # sys.modules back door to a banned module
    "import sys\nm = sys.modules['subprocess']",    # same, without a banned downstream attr
]

MUST_ACCEPT = [
    "import numpy as np\nimport pandas as pd\nx = np.zeros(3)",
    "rets = panel.pct_change()\nw = rets.rolling(63).std()",
    "df = pd.read_parquet(path)",                       # local read via adapter path: fine
    "pd.read_csv(local_path)",                          # non-literal arg: allowed (adapters pass paths)
    "from sdk.adapters import sep_panel, us_universe",
    "mom = panel.pct_change(126)\nsig = np.sign(mom)",
    "result = df.groupby('sector').rank(pct=True)",
    # benign reflection the 162-strategy corpus actually uses -- must STAY accepted:
    "attrs = getattr(panel, 'attrs', {})",          # the dominant legit getattr pattern
    "cols = getattr(df, 'columns', [])",
    "obs = type(e).__name__",                        # benign introspection (7x in corpus)
    "doc = (fn.__doc__ or '').strip()",             # benign read-only metadata
]


@pytest.mark.parametrize("code", MUST_REJECT)
def test_rejects(code):
    assert scan_code(code) is not None, f"should reject: {code!r}"


@pytest.mark.parametrize("code", MUST_ACCEPT)
def test_accepts(code):
    assert scan_code(code) is None, f"should accept: {code!r}"
