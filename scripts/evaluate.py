import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


def _norm_text(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s)


def _norm_email(s) -> str:
    return _norm_text(s)


def _norm_phone(s) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\D", "", str(s))


def _norm_name(s) -> str:
    return _norm_text(s)


def _lcs_length(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(n):
        curr = [0] * (m + 1)
        for j in range(m):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev = curr
    return prev[m]


def _rouge_l(hyp: str, ref: str) -> float:
    if not ref or not hyp:
        return 0.0
    h_toks = hyp.split()
    r_toks = ref.split()
    if not h_toks or not r_toks:
        return 0.0
    lcs = _lcs_length(h_toks, r_toks)
    prec = lcs / len(h_toks)
    rec = lcs / len(r_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _match_persons(gold_df: pd.DataFrame, pred_df: pd.DataFrame):
    gold_names = {_norm_name(r["person_full_name"]): r for _, r in gold_df.iterrows()}
    pred_names = {_norm_name(r["person_full_name"]): r for _, r in pred_df.iterrows()}

    matched, fn, fp = [], [], []
    for name, gold_row in gold_names.items():
        if name in pred_names:
            matched.append((gold_row, pred_names[name]))
        else:
            fn.append(gold_row)
    for name, pred_row in pred_names.items():
        if name not in gold_names:
            fp.append(pred_row)
    return matched, fn, fp


EXACT_FIELDS = ["person_email", "organization_email"]
PHONE_FIELDS = ["person_phone", "organization_phone"]
ROUGE_FIELDS = [
    "position",
    "division_name",
    "address",
    "roiv_full_name",
    "person_bio",
    "organization_phone",
]


def _field_scores(matched_pairs):
    results = {}

    for field in EXACT_FIELDS + PHONE_FIELDS + ROUGE_FIELDS:
        scores, total, covered = [], 0, 0
        for gold_row, pred_row in matched_pairs:
            gold_val = gold_row.get(field, "")
            pred_val = pred_row.get(field, "")
            if pd.isna(gold_val):
                gold_val = ""
            else:
                gold_val = str(gold_val)
            if pd.isna(pred_val):
                pred_val = ""
            else:
                pred_val = str(pred_val)

            if not gold_val.strip():
                continue
            total += 1
            if pred_val.strip():
                covered += 1

            if field in EXACT_FIELDS:
                score = float(_norm_email(gold_val) == _norm_email(pred_val))
            elif field in PHONE_FIELDS:
                gp, pp = _norm_phone(gold_val), _norm_phone(pred_val)
                score = float(gp[-7:] == pp[-7:]) if gp and pp else 0.0
            else:  # ROUGE-L
                score = _rouge_l(_norm_text(pred_val), _norm_text(gold_val))

            scores.append(score)

        results[field] = {
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "coverage": round(covered / total, 4) if total else None,
            "n_gold": total,
        }
    return results


def evaluate(gold_path: str, pred_paths: list[str]):
    gold_df = pd.read_csv(gold_path, dtype=str)

    all_results = []

    for pred_path in pred_paths:
        pred_df = pd.read_csv(pred_path, dtype=str)

        matched, fn, fp = _match_persons(gold_df, pred_df)

        tp = len(matched)
        precision = tp / (tp + len(fp)) if (tp + len(fp)) else 0.0
        recall = tp / (tp + len(fn)) if (tp + len(fn)) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        field_scores = _field_scores(matched)

        all_results.append(
            {
                "file": Path(pred_path).name,
                "gold_persons": len(gold_df),
                "pred_persons": len(pred_df),
                "TP": tp,
                "FP": len(fp),
                "FN": len(fn),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "fields": field_scores,
            }
        )

    _print_report(all_results)
    return all_results


def _print_report(results):
    print()
    print("=" * 70)
    print("  PERSON EXTRACTION EVALUATION")
    print("=" * 70)

    for r in results:
        print(f"\n── {r['file']} ──")
        print(f"  Gold persons : {r['gold_persons']}")
        print(f"  Pred persons : {r['pred_persons']}")
        print(f"  TP={r['TP']}  FP={r['FP']}  FN={r['FN']}")
        print(f"  Precision : {r['precision']:.4f}")
        print(f"  Recall    : {r['recall']:.4f}")
        print(f"  F1        : {r['f1']:.4f}")

        print(f"\n  {'Field':<25} {'Score':>8}  {'Coverage':>9}  {'Gold n':>7}")
        print(f"  {'-' * 25} {'-' * 8}  {'-' * 9}  {'-' * 7}")
        for field, vals in r["fields"].items():
            score = (
                f"{vals['avg_score']:.4f}"
                if vals["avg_score"] is not None
                else "  —   "
            )
            coverage = (
                f"{vals['coverage']:.4f}" if vals["coverage"] is not None else "  —   "
            )
            n = str(vals["n_gold"])
            print(f"  {field:<25} {score:>8}  {coverage:>9}  {n:>7}")

    if len(results) > 1:
        print("\n── MACRO AVERAGE ──")
        for metric in ("precision", "recall", "f1"):
            vals = [r[metric] for r in results]
            print(f"  {metric:<12} {sum(vals) / len(vals):.4f}")

    print()


if __name__ == "__main__":
    import glob

    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="Path to golden dataset CSV")
    p.add_argument(
        "--pred", required=True, nargs="+", help="Path(s) to predicted CSV(s), glob ok"
    )
    args = p.parse_args()

    pred_paths = []
    for pat in args.pred:
        expanded = glob.glob(pat)
        pred_paths.extend(expanded if expanded else [pat])
    pred_paths = [f for f in pred_paths if Path(f).exists()]

    if not pred_paths:
        print("No prediction files found.", file=sys.stderr)
        sys.exit(1)

    evaluate(args.gold, pred_paths)
