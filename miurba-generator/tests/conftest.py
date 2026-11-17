"""Shared test setup: dummy endpoint env vars.

The default example config references ${OPENAI_API_ENDPOINT/MODEL/KEY};
tests that load it only need the keys to expand, never real values.
"""

import os

for _var in ("OPENAI_API_ENDPOINT", "OPENAI_API_MODEL", "OPENAI_API_KEY"):
    os.environ.setdefault(_var, "dummy")
