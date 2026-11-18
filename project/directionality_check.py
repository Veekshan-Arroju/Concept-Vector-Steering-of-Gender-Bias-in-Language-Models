"""
Judge-free, model-free check on a saved stance_logit_test.py result:
does +epsilon move the stance score the OPPOSITE way from -epsilon (a concept direction), or the same way (disruption)?

    python directionality_check.py results/<slug>/<experiment>/stance_logit__<vector>__<tag>.pkl [more.pkl ...]

Reads only the saved table (pickle + numpy): no GPU, no torch.
"""
import argparse
import pickle

from coef_calibration import directionality, format_directionality


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--key", default="stance_shift", help="column of the table to test (default stance_shift)")
    args = ap.parse_args()
    for f in args.files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        table = d["table"] if isinstance(d, dict) and "table" in d else d
        print(f"\n{f}")
        print(format_directionality(directionality(table, key=args.key)))


if __name__ == "__main__":
    main()
