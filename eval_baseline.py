"""eval_baseline.py — measure today's pipeline against v2 invariants.

Walks ``eval/`` for ``smart_ocr_response_*.json`` files, finds the matching
PDF (``test_*.pdf`` in repo root), runs ``cell_grid_service_v4`` to get
grids, adapts the v1 response into the v2 shape, and runs ``cell_layout.validate``.

Output:
  Per-PDF Telemetry summary, then aggregate over the corpus. We use the
  aggregate as the BASELINE — v2 has to beat these numbers (per-invariant)
  to ship.

The PDF↔response pairing convention: a response file named
``smart_ocr_response_FOO.json`` is paired with a PDF file ``test_FOO.pdf``
(or ``FOO.pdf``) in the repository root. Override pairing with --pairs.

Usage:
    python3 eval_baseline.py [--eval-dir eval/] [--pdf-dir .]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from cell_grid_service_v4 import analyze_pdf_cell_grid_v4
from cell_layout import adapt_v1_to_v2, validate

INVARIANT_DESCRIPTIONS = {
    "I1": "cell_id out of grid",
    "I2": "comment_range on non-writable cells",
    "I3": "anchor.range out of grid",
    "I4": "score_range outside score_box_range",
    "I5": "comment_font_pts out of [14, 16]",
    "I6": "comment overflows comment_range at chosen font",
    "I7": "two annotations overlap on same page",
    "I8": "exponent_caret missing extra.missing_word",
    "I9": "curly_brace bad extra.side",
}


def _pdf_for_response(response_path: Path, pdf_dir: Path) -> Path | None:
    """Convention: smart_ocr_response_<stem>.json ↔ <stem>.pdf or test_<stem>.pdf."""
    m = re.match(r"smart_ocr_response_(.+)\.json$", response_path.name)
    if not m:
        return None
    stem = m.group(1)
    for cand in (f"{stem}.pdf", f"test_{stem}.pdf"):
        p = pdf_dir / cand
        if p.exists():
            return p
    return None


def _eval_one(pdf_path: Path, response_path: Path) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    response = json.loads(response_path.read_text())

    # Use the cell_size_pts from the response's grid meta so we match what
    # the production pipeline produced; fall back to v4 default if missing.
    metas = (response.get("data") or {}).get("cellGridMeta") or []
    cs = None
    for m in metas:
        try:
            cs = float(m.get("cell_size_pts"))
            if cs > 0:
                break
        except (TypeError, ValueError):
            continue

    t0 = time.perf_counter()
    grids = (
        analyze_pdf_cell_grid_v4(pdf_bytes, cell_size_pts=cs)
        if cs else analyze_pdf_cell_grid_v4(pdf_bytes)
    )
    grid_secs = time.perf_counter() - t0

    adapted = adapt_v1_to_v2(response)
    t0 = time.perf_counter()
    tel = validate(adapted, grids)
    val_secs = time.perf_counter() - t0

    return {
        "pdf": pdf_path.name,
        "response": response_path.name,
        "grid_secs": grid_secs,
        "validate_secs": val_secs,
        "items_total": tel.items_total,
        "annotations_total": tel.annotations_total,
        "counts": dict(tel.counts),
        "violation_rate": tel.violation_rate(),
        "violations_sample": [v.to_dict() for v in tel.violations[:5]],
    }


def _print_pdf_report(rep: dict) -> None:
    print(f"\n── {rep['pdf']}  vs  {rep['response']}", file=sys.stderr)
    print(
        f"   items={rep['items_total']}  annotations={rep['annotations_total']}  "
        f"grid={rep['grid_secs']:.1f}s  validate={1000*rep['validate_secs']:.0f}ms",
        file=sys.stderr,
    )
    if rep["counts"]:
        for inv, n in sorted(rep["counts"].items()):
            print(f"   {inv}={n:<3} {INVARIANT_DESCRIPTIONS.get(inv,'')}", file=sys.stderr)
    else:
        print("   no violations", file=sys.stderr)
    for v in rep["violations_sample"]:
        print(f"     · {v['invariant']} p{v['page']} ann#{v['annotation_index']}: "
              f"{v['detail']}", file=sys.stderr)


def _print_aggregate(reports: list[dict]) -> dict:
    if not reports:
        return {}
    total_anns = sum(r["annotations_total"] for r in reports)
    total_items = sum(r["items_total"] for r in reports)
    agg = Counter()
    for r in reports:
        for k, v in r["counts"].items():
            agg[k] += v

    print("\n────────────────────────────────────────────────", file=sys.stderr)
    print(f"AGGREGATE  pdfs={len(reports)}  items={total_items}  "
          f"annotations={total_anns}", file=sys.stderr)
    print("────────────────────────────────────────────────", file=sys.stderr)
    if not agg:
        print("  no violations across corpus  ✓", file=sys.stderr)
    else:
        for inv in sorted(agg.keys()):
            n = agg[inv]
            pct = 100.0 * n / max(1, total_anns)
            print(
                f"  {inv}  {n:>3} ({pct:5.1f}%)  {INVARIANT_DESCRIPTIONS.get(inv,'')}",
                file=sys.stderr,
            )

    overall_rate = sum(agg.values()) / max(1, total_anns)
    print(f"\n  overall violation rate: {overall_rate*100:.1f}%  "
          f"(repair gate at promotion: <5%)", file=sys.stderr)
    print(
        "\n  CAVEAT: this validates today's responses against the CURRENT\n"
        "  cell_grid_service_v4 thresholds. Grid drift between request-time\n"
        "  and validation-time inflates I2/I6 hits. Use this baseline as a\n"
        "  REGRESSION CEILING, not as ground truth — v2 must beat these\n"
        "  numbers because v2 ships the grid alongside the response (no drift).",
        file=sys.stderr,
    )
    return {
        "pdfs": len(reports),
        "items_total": total_items,
        "annotations_total": total_anns,
        "counts": dict(agg),
        "overall_violation_rate": overall_rate,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default="eval", type=Path)
    p.add_argument("--pdf-dir", default=".", type=Path)
    p.add_argument("--json", action="store_true",
                   help="Emit JSON report on stdout")
    args = p.parse_args()

    response_paths = sorted(args.eval_dir.glob("smart_ocr_response_*.json"))
    if not response_paths:
        print(f"no response files found in {args.eval_dir}", file=sys.stderr)
        return 1

    reports: list[dict] = []
    for rp in response_paths:
        pdf = _pdf_for_response(rp, args.pdf_dir)
        if pdf is None:
            print(f"  skip: no matching PDF for {rp.name}", file=sys.stderr)
            continue
        rep = _eval_one(pdf, rp)
        _print_pdf_report(rep)
        reports.append(rep)

    agg = _print_aggregate(reports)
    if args.json:
        json.dump(
            {"pdfs": reports, "aggregate": agg},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
