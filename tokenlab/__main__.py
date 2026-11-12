"""python -m tokenlab [data|backtest|calibrate|stress|report|all]"""
import sys

from . import calibrate, report, stress
from .backtest import data, study

STEPS = {"data": data.build, "backtest": study.run, "calibrate": calibrate.run,
         "stress": stress.run, "report": report.build}


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "all"
    if cmd != "all" and cmd not in STEPS:
        print(__doc__)
        return 2
    for step in (["backtest", "calibrate", "stress", "report"] if cmd == "all" else [cmd]):
        print(f"== {step} ==")
        STEPS[step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
