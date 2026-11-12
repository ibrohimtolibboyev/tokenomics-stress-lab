"""Command line entry point.

    python -m tokenlab data       # rebuild the real unlock sample (needs internet)
    python -m tokenlab backtest   # event study on the 60 unlocks
    python -m tokenlab calibrate  # fit market depth to the backtest
    python -m tokenlab stress     # 12 designs x 6 scenarios + sensitivity
    python -m tokenlab report     # figures + REPORT.md
    python -m tokenlab all        # backtest -> calibrate -> stress -> report (offline)
"""
from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "all"
    if cmd == "data":
        from .backtest import data
        data.build()
    elif cmd in ("backtest", "calibrate", "stress", "report", "all"):
        steps = ["backtest", "calibrate", "stress", "report"] if cmd == "all" else [cmd]
        for step in steps:
            print(f"== {step} ==")
            if step == "backtest":
                from .backtest import study
                study.run()
            elif step == "calibrate":
                from . import calibrate
                calibrate.run()
            elif step == "stress":
                from . import stress
                stress.run()
            else:
                from . import report
                report.build()
                print("wrote REPORT.md")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
