"""Precompute the default run: data_edited.csv -> cached pipeline artifacts.

Usage:
    python scripts/build_default_run.py [--force]

The Streamlit app auto-loads the cache on startup; run this after pulling
or when data_edited.csv changes. Takes a while: it trains every segmentation
model (hierarchical Ward linkage and the consensus ensemble are O(n^2) at ~25k
customers) and every churn model on ~25k customers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.bootstrap_service import build_default_run


def main() -> int:
    import logging
    from datetime import datetime

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"logs/build_default_run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
                encoding="utf-8",
            ),
        ],
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if fresh.")
    args = parser.parse_args()
    manifest = build_default_run(force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
