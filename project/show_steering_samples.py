"""
Print what steering_eval.py saved: a per-coefficient summary and chosen generations.

    python show_steering_samples.py results/<slug>/<experiment>/steering_eval__pooled__misogyny__mytag.csv --prompts 3 --coefs -3 3 --chars 300

The summary is recomputed from the CSV, including the judge-free degeneration check, so it also works on files saved before
that check existed. `degenerate` = share of generations that collapsed into repetition (judge.degeneration_heuristic).
"""

import argparse
import csv

from judge import degeneration_heuristic, refusal_heuristic


def load_rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["coef"] = float(r["coef"])
        r["degenerate"] = degeneration_heuristic(r["response"])
        r["refusal"] = int(r.get("refusal_heuristic") or refusal_heuristic(r["response"]))
    return rows


def coef_summary(rows: list) -> list:
    out = []
    for c in sorted({r["coef"] for r in rows}):
        sub = [r for r in rows if r["coef"] == c]
        out.append({"coef": c, "n": len(sub), "degenerate": sum(r["degenerate"] for r in sub) / len(sub),
                    "refusal": sum(r["refusal"] for r in sub) / len(sub)})
    return out


def print_summary(summary: list) -> None:
    print(f"\n  {'coef':>6}  {'n':>3}  {'degenerate':>10}  {'refusal kw':>10}")
    for e in summary:
        print(f"  {e['coef']:+6.2f}  {e['n']:3d}  {e['degenerate']:10.2f}  {e['refusal']:10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show saved steering generations.")
    parser.add_argument("csv")
    parser.add_argument("--prompts", type=int, default=2, help="how many prompts to print (0 = summary only)")
    parser.add_argument("--coefs", nargs="*", type=float, default=None, help="only these coefficients")
    parser.add_argument("--chars", type=int, default=300, help="characters of each response to print")
    args = parser.parse_args()
    rows = load_rows(args.csv)
    print_summary(coef_summary(rows))
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))[:args.prompts]
    for p in prompts:
        print("\n###", p)
        for r in rows:
            if r["prompt"] == p and (args.coefs is None or r["coef"] in args.coefs):
                flag = " DEGENERATE" if r["degenerate"] else ""
                print(f"\n[coef {r['coef']:+.1f}{flag}] {r['response'][:args.chars]}")


if __name__ == "__main__":
    main()
