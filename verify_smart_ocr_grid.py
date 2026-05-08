"""Verify that smart-ocr API cell IDs match cell_grid_service_v4 output for a PDF.

Run:
    python3 verify_smart_ocr_grid.py test_pdf4.pdf smart_ocr_response_pdf4.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cell_grid_service_v4 import analyze_pdf_cell_grid_v4, rc_from_cell_id


def cell_bbox_percent(meta: dict, cell_id: str) -> tuple[float, float, float, float]:
    """Compute (x1%, y1%, x2%, y2%) of a single cell from grid meta."""
    row, col = rc_from_cell_id(cell_id)
    cs = meta["cell_size_pts"]
    lm = meta["left_margin_pts"]
    tm = meta["top_margin_pts"]
    pw = meta["page_w_pts"]
    ph = meta["page_h_pts"]
    x1 = lm + (col - 1) * cs
    x2 = lm + col * cs
    y1 = tm + (row - 1) * cs
    y2 = tm + row * cs
    return (
        round(100.0 * x1 / pw, 2),
        round(100.0 * y1 / ph, 2),
        round(100.0 * x2 / pw, 2),
        round(100.0 * y2 / ph, 2),
    )


def cells_to_bbox(meta: dict, cell_ids: list[str]) -> dict:
    rows_cols = [rc_from_cell_id(c) for c in cell_ids]
    rs = [r for r, _ in rows_cols]
    cs = [c for _, c in rows_cols]
    r1, r2 = min(rs), max(rs)
    c1, c2 = min(cs), max(cs)
    cell_pt = meta["cell_size_pts"]
    lm = meta["left_margin_pts"]
    tm = meta["top_margin_pts"]
    pw = meta["page_w_pts"]
    ph = meta["page_h_pts"]
    x1 = lm + (c1 - 1) * cell_pt
    x2 = lm + c2 * cell_pt
    y1 = tm + (r1 - 1) * cell_pt
    y2 = tm + r2 * cell_pt
    return {
        "x1_percent": round(100.0 * x1 / pw, 2),
        "y1_percent": round(100.0 * y1 / ph, 2),
        "x2_percent": round(100.0 * x2 / pw, 2),
        "y2_percent": round(100.0 * y2 / ph, 2),
    }


def main(pdf_path: str, response_json_path: str) -> int:
    pdf = Path(pdf_path).read_bytes()
    api = json.loads(Path(response_json_path).read_text())
    api_meta = api["data"]["cellGridMeta"]

    print(f"Running cell_grid_service_v4 on {pdf_path} ...")
    grids = analyze_pdf_cell_grid_v4(pdf)
    svc_meta = [
        {
            "page": g.page,
            "rows": g.rows,
            "cols": g.cols,
            "cell_size_pts": g.cell_size_pts,
            "left_margin_pts": g.left_margin_pts,
            "top_margin_pts": g.top_margin_pts,
            "page_w_pts": g.page_w_pts,
            "page_h_pts": g.page_h_pts,
        }
        for g in grids
    ]

    # ── Section 1: Compare per-page grid meta ─────────────────────────────
    print("\n=== Per-page grid meta comparison (API vs service) ===")
    if len(api_meta) != len(svc_meta):
        print(f"!! Page count mismatch: api={len(api_meta)} svc={len(svc_meta)}")
    pages = min(len(api_meta), len(svc_meta))
    fields = ["rows", "cols", "cell_size_pts", "left_margin_pts",
              "top_margin_pts", "page_w_pts", "page_h_pts"]
    meta_mismatches = 0
    for i in range(pages):
        a = api_meta[i]
        s = svc_meta[i]
        diffs = [f for f in fields if a.get(f) != s.get(f)]
        status = "OK" if not diffs else "DIFF"
        line = f"  page {a['page']:>2}: {status}"
        if diffs:
            meta_mismatches += 1
            line += "  " + ", ".join(
                f"{f}: api={a[f]} svc={s[f]}" for f in diffs
            )
        print(line)
    print(f"  → {pages - meta_mismatches}/{pages} pages match exactly")

    # Build lookup of svc grids by page number for bbox check
    svc_by_page = {g["page"]: g for g in svc_meta}

    # ── Section 2: Verify annotation bbox derivation ──────────────────────
    print("\n=== Annotation cell_ids → bbox check (using SERVICE meta) ===")
    total = 0
    bbox_match = 0
    for item in api["data"]["items"]:
        for ann in item.get("annotations", []):
            if not ann.get("cell_ids"):
                continue
            total += 1
            bbox = ann["bbox"]
            page = bbox["page"]
            meta = svc_by_page.get(page)
            if not meta:
                print(f"  q{item['question_id']} page {page}: no svc grid")
                continue
            recomputed = cells_to_bbox(meta, ann["cell_ids"])
            api_box = {k: bbox[k] for k in
                       ("x1_percent", "y1_percent", "x2_percent", "y2_percent")}
            ok = all(abs(recomputed[k] - api_box[k]) < 0.05 for k in api_box)
            if ok:
                bbox_match += 1
            else:
                print(f"  q{item['question_id']} page {page} {ann['range_id']}:")
                print(f"     api bbox  = {api_box}")
                print(f"     recomputed = {recomputed}")
    print(f"  → {bbox_match}/{total} annotation bboxes reproduce exactly "
          f"from cell_ids using service grid meta")

    # ── Section 3: Sanity-check cell IDs against grid bounds ──────────────
    print("\n=== Cell ID range sanity (within rows×cols of svc grid) ===")
    out_of_range = 0
    for item in api["data"]["items"]:
        for ann in item.get("annotations", []):
            if not ann.get("cell_ids"):
                continue
            page = ann["bbox"]["page"]
            meta = svc_by_page.get(page)
            if not meta:
                continue
            for cid in ann["cell_ids"]:
                r, c = rc_from_cell_id(cid)
                if r < 1 or r > meta["rows"] or c < 1 or c > meta["cols"]:
                    out_of_range += 1
                    print(f"  q{item['question_id']} page {page}: "
                          f"{cid} (r={r},c={c}) out of {meta['rows']}x{meta['cols']}")
    print(f"  → {out_of_range} out-of-range cell IDs")

    print("\n=== Summary ===")
    print(f"  meta pages matching: {pages - meta_mismatches}/{pages}")
    print(f"  bboxes matching:     {bbox_match}/{total}")
    print(f"  out-of-range cells:  {out_of_range}")
    return 0 if (meta_mismatches == 0 and out_of_range == 0) else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
