"""
How much do two judges agree? Reads two *__judged.csv files written by judge_csv.py (any two judges / models), matches the rows that
appear in both with a verdict in both, and prints, per criterion, the agreement and Cohen's kappa (agreement beyond chance).

    python compare_judged.py judged_local/offtopic_oldv3__judged.csv judged/offtopic_oldv3__judged.csv
    python compare_judged.py A__stems.csv B__stems.csv --criteria stereotype demeaning coherent      (files written by stereotype_eval.py)

Rule of thumb: kappa above ~0.6 on `misogynistic` means the cheaper judge can be used for the comparison between vectors.
"""
import sys

import pandas as pd

CRITERIA = ("misogynistic", "coherent", "refusal_or_hedge")


def kappa(a, b) -> float:
    a, b = list(map(int, a)), list(map(int, b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compare(a: pd.DataFrame, b: pd.DataFrame, criteria=CRITERIA) -> pd.DataFrame:
    key = ["direction", "coef", "prompt", "response"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    rows = []
    for c in criteria:
        ok = m.dropna(subset=[c + "_a", c + "_b"])
        n = len(ok)
        agree = float((ok[c + "_a"].astype(int) == ok[c + "_b"].astype(int)).mean()) if n else float("nan")
        both1 = int(((ok[c + "_a"].astype(int) == 1) & (ok[c + "_b"].astype(int) == 1)).sum()) if n else 0
        rows.append({"criterion": c, "n_rows_both": n, "agreement": round(agree, 3), "kappa": round(kappa(ok[c + "_a"], ok[c + "_b"]), 3) if n else float("nan"),
                     "positive_in_A": int(ok[c + "_a"].astype(int).sum()) if n else 0, "positive_in_B": int(ok[c + "_b"].astype(int).sum()) if n else 0,
                     "both_positive": both1})
    return pd.DataFrame(rows)


def main(argv) -> None:
    criteria = CRITERIA
    if "--criteria" in argv:
        i = argv.index("--criteria")
        criteria, argv = tuple(argv[i + 1:]), argv[:i]
        if not criteria:
            raise SystemExit("--criteria needs at least one column name")
    if len(argv) != 2:
        raise SystemExit(__doc__)
    a, b = pd.read_csv(argv[0]), pd.read_csv(argv[1])
    print(compare(a, b, criteria).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
