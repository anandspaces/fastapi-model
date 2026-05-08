#!/usr/bin/env python3
"""
Regression + smoke tests for teacher_annotate.py

Run:  python3 test_teacher_annotate.py
Exits with non-zero status if any test fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import Callable

# Local import — keep path explicit so test runner doesn't need pytest config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_annotate import (
    build_layout_map, load_pdf_pages_pil, scaffold_dense,
    RemarkResolver, ProximityLayout, process_pdf,
)

DPI = 250

# Ground truth — values measured from annotator's own detect_lines/detect_gaps
# at DPI=250. Tolerance: line count must match exactly; gap height ±20%.
KNOWN_LAYOUTS: dict[str, dict[int, dict]] = {
    "test_original_pdf1.pdf": {
        1: {"lines": 22, "gaps": [(13, 154), (21, 92)]},
        2: {"lines": 19, "gaps": [(3, 140), (17, 118), (18, 245)]},
    },
    "test_original_pdf2.pdf": {
        1: {"lines": 18, "gaps": [(4, 107), (6, 138), (9, 92)]},
        2: {"lines": 20, "gaps": [(19, 201)]},
    },
}

GAP_HEIGHT_TOL = 0.20      # 20%
SMOKE_OUTPUT_MIN_BYTES = 100_000     # 100 KB


# ─────────────────────────────────────────────────────────────────────────────
# Tiny test runner — no pytest dependency
# ─────────────────────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed: list[tuple[str, str]] = []

    def run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            self.passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            self.failed.append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception:
            self.failed.append((name, traceback.format_exc()))
            print(f"  ✗ {name}: {traceback.format_exc()}")

    def report(self) -> int:
        total = self.passed + len(self.failed)
        print(f"\n{self.passed}/{total} passed")
        if self.failed:
            print("\nFailures:")
            for name, msg in self.failed:
                print(f"\n  {name}:\n    {msg.strip()}")
            return 1
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Layout regression — assert build_layout_map matches KNOWN_LAYOUTS
# ─────────────────────────────────────────────────────────────────────────────

def _assert_layout(pdf_name: str) -> None:
    expected = KNOWN_LAYOUTS[pdf_name]
    pages = load_pdf_pages_pil(pdf_name, DPI)
    assert len(pages) >= max(expected.keys()), \
        f"{pdf_name}: only {len(pages)} pages, expected ≥{max(expected.keys())}"
    for page_no, exp in expected.items():
        L = build_layout_map(pages[page_no - 1], page_no, DPI)
        assert len(L.line_ys) == exp["lines"], (
            f"{pdf_name} p{page_no}: line count {len(L.line_ys)} != {exp['lines']}")
        actual_gaps = [(g.after_line, g.height) for g in L.gaps]
        # Must contain every expected gap (within tolerance) — extra small gaps are OK
        for exp_after, exp_h in exp["gaps"]:
            match = next((g for g in actual_gaps if g[0] == exp_after), None)
            assert match is not None, (
                f"{pdf_name} p{page_no}: gap after_line={exp_after} not found "
                f"(got {actual_gaps})")
            ratio = match[1] / exp_h
            assert (1 - GAP_HEIGHT_TOL) <= ratio <= (1 + GAP_HEIGHT_TOL), (
                f"{pdf_name} p{page_no}: gap after_line={exp_after} height "
                f"{match[1]} not within ±20% of expected {exp_h}")


def test_layout_pdf1(): _assert_layout("test_original_pdf1.pdf")
def test_layout_pdf2(): _assert_layout("test_original_pdf2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Word gaps + free rects + score position
# ─────────────────────────────────────────────────────────────────────────────

def test_word_gaps_present():
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    L = build_layout_map(pages[0], 1, DPI)
    assert len(L.word_gaps) >= 5, f"only {len(L.word_gaps)} word gaps detected"
    # Every gap should fall within the body band
    for wg in L.word_gaps:
        assert L.lmx <= wg.x_center <= L.rmx, \
            f"word gap x_center={wg.x_center} outside [{L.lmx},{L.rmx}]"


def test_score_position_resolves():
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    L = build_layout_map(pages[0], 1, DPI)
    cx_pct, cy_line = L.best_score_position()
    assert 0 <= cx_pct <= 100
    assert 1 <= cy_line <= len(L.line_ys)


# ─────────────────────────────────────────────────────────────────────────────
# Resolver — auto / fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_resolver_auto_score():
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    L = build_layout_map(pages[0], 1, DPI)
    raw = [{"type": "score_circle", "cx": "auto", "cy_line": "auto",
            "text": "8/10", "severity": "critical"}]
    resolved = RemarkResolver(L).resolve_all(raw)
    assert len(resolved) == 1
    r = resolved[0]
    assert 0 < r.cx_px < L.W
    assert 0 < r.cy_px < L.H
    assert r.text == "8/10"


def test_resolver_gap_fallback():
    """If remark_zone='gap' but no gap within ±5 lines → fall back to right_margin."""
    pages = load_pdf_pages_pil("test_original_pdf2.pdf", DPI)
    L = build_layout_map(pages[1], 2, DPI)   # pdf2 page 2: only gap is after_line=19
    raw = [{
        "type": "underline_remark", "line": 5,    # too far from after_line=19
        "x_start": 14, "x_end": 60,
        "remark": "x", "remark_zone": "gap", "severity": "critical",
    }]
    resolved = RemarkResolver(L).resolve_all(raw)
    assert len(resolved) == 1
    assert resolved[0].remark_zone == "right_margin", \
        f"expected fallback, got remark_zone={resolved[0].remark_zone}"


# ─────────────────────────────────────────────────────────────────────────────
# ProximityLayout — pre-seeded slots prevent overlap with occupied margin
# ─────────────────────────────────────────────────────────────────────────────

def test_proximity_pre_seeded():
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    L = build_layout_map(pages[0], 1, DPI)
    pl = ProximityLayout(L)
    # If right strips were detected, there must be occupied ranges seeded
    if L.right_strips:
        assert len(pl._slots["right_margin"]) > 0, \
            "right_margin should have pre-seeded occupied slots"


def test_proximity_no_overlap():
    """Stack 8 remarks in right_margin → resulting slots are pairwise non-overlapping."""
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    L = build_layout_map(pages[0], 1, DPI)
    pl = ProximityLayout(L)
    h = 60
    placed_ys = [pl.find_y("right_margin", L.line_ys[i] if i < len(L.line_ys) else 100, h)
                 for i in range(8)]
    # Sort and check every pair
    intervals = sorted([(y, y + h) for y in placed_ys])
    for i in range(len(intervals) - 1):
        assert intervals[i][1] <= intervals[i + 1][0] + 1, \
            f"slots overlap: {intervals[i]} and {intervals[i+1]}"


# ─────────────────────────────────────────────────────────────────────────────
# scaffold_dense — every page produces ≥ 5 remarks
# ─────────────────────────────────────────────────────────────────────────────

def test_scaffold_dense_pdf1():
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    for i, img in enumerate(pages, 1):
        L = build_layout_map(img, i, DPI)
        spec = scaffold_dense(L)
        assert len(spec) >= 5, f"pdf1 p{i}: only {len(spec)} dense remarks"
        # Score circle must be present and first by priority
        types = [r["type"] for r in spec]
        assert "score_circle" in types
        assert "tick" in types


def test_scaffold_dense_pdf2():
    pages = load_pdf_pages_pil("test_original_pdf2.pdf", DPI)
    for i, img in enumerate(pages, 1):
        L = build_layout_map(img, i, DPI)
        spec = scaffold_dense(L)
        assert len(spec) >= 4, f"pdf2 p{i}: only {len(spec)} dense remarks"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end smoke tests — render PDF, assert it exists and is non-trivial
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_render(pdf_in: str, remarks_path: str, label: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, f"{label}.pdf")
        process_pdf(pdf_in, remarks_path, out, dpi=DPI)
        assert os.path.exists(out), f"{label}: output PDF missing"
        size = os.path.getsize(out)
        assert size >= SMOKE_OUTPUT_MIN_BYTES, \
            f"{label}: output too small ({size} bytes)"


def test_render_smoke_pdf1():
    _smoke_render("test_original_pdf1.pdf", "test_remarks_pdf1.json", "pdf1")


def test_render_smoke_pdf2():
    _smoke_render("test_original_pdf2.pdf", "test_remarks_pdf2.json", "pdf2")


def test_render_dense_pdf1():
    """scaffold_dense → write JSON → render → output exists."""
    import json
    pages = load_pdf_pages_pil("test_original_pdf1.pdf", DPI)
    spec = {"pages": [{"page": i + 1,
                       "remarks": scaffold_dense(build_layout_map(p, i + 1, DPI))}
                      for i, p in enumerate(pages)]}
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = os.path.join(tmp, "dense.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        out = os.path.join(tmp, "dense.pdf")
        process_pdf("test_original_pdf1.pdf", spec_path, out, dpi=DPI)
        assert os.path.exists(out)
        assert os.path.getsize(out) >= SMOKE_OUTPUT_MIN_BYTES


def test_render_dense_pdf2():
    import json
    pages = load_pdf_pages_pil("test_original_pdf2.pdf", DPI)
    spec = {"pages": [{"page": i + 1,
                       "remarks": scaffold_dense(build_layout_map(p, i + 1, DPI))}
                      for i, p in enumerate(pages)]}
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = os.path.join(tmp, "dense.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        out = os.path.join(tmp, "dense.pdf")
        process_pdf("test_original_pdf2.pdf", spec_path, out, dpi=DPI)
        assert os.path.exists(out)
        assert os.path.getsize(out) >= SMOKE_OUTPUT_MIN_BYTES


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    runner = TestRunner()
    print("Running teacher_annotate.py regression suite...\n")

    print("[layout regression]")
    runner.run("layout_pdf1", test_layout_pdf1)
    runner.run("layout_pdf2", test_layout_pdf2)

    print("\n[detection]")
    runner.run("word_gaps_present", test_word_gaps_present)
    runner.run("score_position_resolves", test_score_position_resolves)

    print("\n[resolver]")
    runner.run("resolver_auto_score", test_resolver_auto_score)
    runner.run("resolver_gap_fallback", test_resolver_gap_fallback)

    print("\n[proximity layout]")
    runner.run("proximity_pre_seeded", test_proximity_pre_seeded)
    runner.run("proximity_no_overlap", test_proximity_no_overlap)

    print("\n[scaffold_dense]")
    runner.run("scaffold_dense_pdf1", test_scaffold_dense_pdf1)
    runner.run("scaffold_dense_pdf2", test_scaffold_dense_pdf2)

    print("\n[end-to-end smoke (slow)]")
    runner.run("render_smoke_pdf1", test_render_smoke_pdf1)
    runner.run("render_smoke_pdf2", test_render_smoke_pdf2)
    runner.run("render_dense_pdf1", test_render_dense_pdf1)
    runner.run("render_dense_pdf2", test_render_dense_pdf2)

    return runner.report()


if __name__ == "__main__":
    sys.exit(main())
