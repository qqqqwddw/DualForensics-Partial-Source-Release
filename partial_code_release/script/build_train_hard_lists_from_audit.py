import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_VIEWS = (
    "social_chain_strong",
    "platform_chain",
    "jpeg_q50",
    "mid_suppress",
)


def parse_views(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_VIEWS)
    return [v.strip() for v in raw.split(",") if v.strip()]


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not fields:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_path_list(path: str, paths: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in paths:
            f.write(str(item) + "\n")


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")


def as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def infer_domain(path: str) -> str:
    norm = os.path.normpath(str(path))
    parts = norm.split(os.sep)
    for idx, part in enumerate(parts):
        if part in ("0_real", "1_fake", "1_false") and idx > 0:
            return parts[idx - 1]
    return parts[-3] if len(parts) >= 3 else "unknown"


def looks_like_test_path(path: str) -> bool:
    parts = [p.lower() for p in os.path.normpath(str(path)).split(os.sep)]
    return "test" in parts


def candidate_key(row: Dict[str, object]) -> Tuple[int, str]:
    return int(row.get("label", -1)), str(row.get("path", ""))


def keep_best_per_path(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[Tuple[int, str], Dict[str, object]] = {}
    for row in rows:
        key = candidate_key(row)
        old = best.get(key)
        if old is None or float(row["score"]) > float(old["score"]):
            best[key] = row
    return list(best.values())


def domain_balanced_select(
    rows: Sequence[Dict[str, object]],
    max_items: int,
    max_per_domain: int,
) -> List[Dict[str, object]]:
    if max_items <= 0:
        return []
    rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    by_domain: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain", "unknown"))].append(row)

    selected: List[Dict[str, object]] = []
    domain_counts: Dict[str, int] = defaultdict(int)
    domain_order = sorted(by_domain.keys(), key=lambda d: len(by_domain[d]), reverse=True)

    progressed = True
    while len(selected) < max_items and progressed:
        progressed = False
        for domain in domain_order:
            if len(selected) >= max_items:
                break
            if max_per_domain > 0 and domain_counts[domain] >= max_per_domain:
                continue
            bucket = by_domain[domain]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            domain_counts[domain] += 1
            progressed = True

    if len(selected) < max_items:
        used = {candidate_key(row) for row in selected}
        for row in rows:
            if len(selected) >= max_items:
                break
            if candidate_key(row) in used:
                continue
            selected.append(row)
            used.add(candidate_key(row))
    return selected


def collect_clean_audit_candidates(
    rows: Sequence[Dict[str, str]],
    detector: str,
    min_score: float,
    allow_test_paths: bool,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    prob_key = f"{detector}_npz_prob"
    pred_key = f"{detector}_npz_pred"
    for row in rows:
        path = str(row.get("path", ""))
        if not path or path.startswith("__"):
            continue
        if (not allow_test_paths) and looks_like_test_path(path):
            continue
        label = as_int(row.get("label"), -1)
        if label not in (0, 1):
            continue
        pred = as_int(row.get(pred_key), -1)
        prob = as_float(row.get(prob_key), 0.0)
        reason = str(row.get("reason", ""))
        is_hard_fake = label == 1 and (pred == 0 or f"{detector}_fn" in reason)
        is_hard_real = label == 0 and (pred == 1 or f"{detector}_fp" in reason)
        if not (is_hard_fake or is_hard_real):
            continue
        score = (1.0 - prob) if label == 1 else prob
        if score < min_score:
            continue
        out.append(
            {
                "path": path,
                "label": label,
                "domain": infer_domain(path),
                "source_view": "clean",
                "source": "clean_audit_error",
                "score": float(score),
                "prob": float(prob),
                "logit_shift": 0.0,
                "clean_pred": pred,
                "perturbed_pred": pred,
                "reason": reason,
            }
        )
    return out


def collect_shift_candidates(
    rows: Sequence[Dict[str, str]],
    detector: str,
    views: Sequence[str],
    min_abs_logit_shift: float,
    allow_test_paths: bool,
) -> List[Dict[str, object]]:
    prob_key = f"{detector}_prob"
    clean_prob_key = f"{detector}_clean_prob"
    pred_key = f"{detector}_pred"
    clean_pred_key = f"{detector}_clean_pred"
    shift_key = f"{detector}_logit_shift"

    out: List[Dict[str, object]] = []
    view_set = set(views)
    for row in rows:
        path = str(row.get("path", ""))
        if not path:
            continue
        if (not allow_test_paths) and looks_like_test_path(path):
            continue
        view = str(row.get("view", ""))
        if view not in view_set:
            continue
        label = as_int(row.get("label"), -1)
        if label not in (0, 1):
            continue

        clean_pred = as_int(row.get(clean_pred_key), -1)
        pred = as_int(row.get(pred_key), -1)
        prob = as_float(row.get(prob_key), 0.0)
        clean_prob = as_float(row.get(clean_prob_key), prob)
        logit_shift = as_float(row.get(shift_key), 0.0)
        abs_shift = abs(logit_shift)

        fake_to_real = label == 1 and pred == 0
        real_to_fake = label == 0 and pred == 1
        fake_weakened = label == 1 and logit_shift < -min_abs_logit_shift and prob < clean_prob
        real_weakened = label == 0 and logit_shift > min_abs_logit_shift and prob > clean_prob
        if not (fake_to_real or real_to_fake or fake_weakened or real_weakened):
            continue

        if label == 1:
            score = (3.0 if fake_to_real else 1.0) + abs_shift + max(0.0, clean_prob - prob)
            source = "perturb_fake_to_real" if fake_to_real else "perturb_fake_score_drop"
        else:
            score = (3.0 if real_to_fake else 1.0) + abs_shift + max(0.0, prob - clean_prob)
            source = "perturb_real_to_fake" if real_to_fake else "perturb_real_score_rise"

        out.append(
            {
                "path": path,
                "label": label,
                "domain": infer_domain(path),
                "source_view": view,
                "source": source,
                "score": float(score),
                "prob": float(prob),
                "clean_prob": float(clean_prob),
                "logit_shift": float(logit_shift),
                "clean_pred": clean_pred,
                "perturbed_pred": pred,
                "reason": source,
            }
        )
    return out


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_domain: Dict[str, int] = defaultdict(int)
    by_source: Dict[str, int] = defaultdict(int)
    by_view: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_domain[str(row.get("domain", "unknown"))] += 1
        by_source[str(row.get("source", "unknown"))] += 1
        by_view[str(row.get("source_view", "unknown"))] += 1
    return {
        "n": len(rows),
        "by_domain": dict(sorted(by_domain.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_view": dict(sorted(by_view.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--audit_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--detector", type=str, default="hos", choices=["hos", "main"])
    parser.add_argument("--views", type=str, default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--max_hard_fake", type=int, default=2000)
    parser.add_argument("--max_hard_real", type=int, default=2000)
    parser.add_argument("--max_per_domain", type=int, default=300)
    parser.add_argument("--min_abs_logit_shift", type=float, default=0.35)
    parser.add_argument("--min_clean_error_score", type=float, default=0.0)
    parser.add_argument("--include_clean_errors", action="store_true")
    parser.add_argument("--allow_test_paths", action="store_true",
                        help="Disabled by default to prevent test leakage into train hard replay.")
    return parser.parse_args()


def main() -> None:
    opt = parse_args()
    views = parse_views(opt.views)
    shift_rows = read_csv_rows(os.path.join(opt.audit_dir, "sample_score_shifts.csv"))
    audit_rows = read_csv_rows(os.path.join(opt.audit_dir, "audit_sample.csv"))
    if not shift_rows and not audit_rows:
        raise RuntimeError(f"No audit outputs found under {opt.audit_dir}")

    candidates = collect_shift_candidates(
        shift_rows,
        detector=opt.detector,
        views=views,
        min_abs_logit_shift=float(opt.min_abs_logit_shift),
        allow_test_paths=bool(opt.allow_test_paths),
    )
    if opt.include_clean_errors:
        candidates.extend(
            collect_clean_audit_candidates(
                audit_rows,
                detector=opt.detector,
                min_score=float(opt.min_clean_error_score),
                allow_test_paths=bool(opt.allow_test_paths),
            )
        )

    candidates = keep_best_per_path(candidates)
    hard_fake = domain_balanced_select(
        [row for row in candidates if int(row["label"]) == 1],
        max_items=int(opt.max_hard_fake),
        max_per_domain=int(opt.max_per_domain),
    )
    hard_real = domain_balanced_select(
        [row for row in candidates if int(row["label"]) == 0],
        max_items=int(opt.max_hard_real),
        max_per_domain=int(opt.max_per_domain),
    )
    selected = hard_fake + hard_real
    selected = sorted(selected, key=lambda r: (int(r["label"]), str(r["domain"]), -float(r["score"])))

    os.makedirs(opt.out_dir, exist_ok=True)
    hard_fake_path = os.path.join(opt.out_dir, f"hard_fake_{opt.detector}_train_only.txt")
    hard_real_path = os.path.join(opt.out_dir, f"hard_real_{opt.detector}_train_only.txt")
    manifest_path = os.path.join(opt.out_dir, f"hard_cases_{opt.detector}_manifest.csv")
    summary_path = os.path.join(opt.out_dir, "summary.json")

    write_path_list(hard_fake_path, [row["path"] for row in hard_fake])
    write_path_list(hard_real_path, [row["path"] for row in hard_real])
    write_csv_rows(manifest_path, selected)
    summary = {
        "audit_dir": opt.audit_dir,
        "detector": opt.detector,
        "views": views,
        "allow_test_paths": bool(opt.allow_test_paths),
        "min_abs_logit_shift": float(opt.min_abs_logit_shift),
        "candidate_total": len(candidates),
        "hard_fake": summarize(hard_fake),
        "hard_real": summarize(hard_real),
        "hard_fake_list": hard_fake_path,
        "hard_real_list": hard_real_path,
        "manifest": manifest_path,
        "note": "Default mode rejects paths containing a test split component to avoid test leakage.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
