"""ASCII occupancy maps for Smart-OCR v2 regression (stage3_block_grade)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cell_grid_service_v4 import PageCellGrid

from .snap import cells_from_range_strings

_BLOCK_PAGE_RE = re.compile(r"^p(\d+)-", re.I)


def _col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _header(cols: int) -> str:
    letters = [_col_letter(i) for i in range(1, cols + 1)]
    return "   " + "".join(f"{c:<2}" for c in letters)


def write_occupancy_map_md(
    out_path: Path,
    grids: list[PageCellGrid],
    flat_blocks: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
) -> None:
    """Write per-page ASCII maps: Q printed, A answer, M marking, * comment, . free."""
    by_page = {g.page: g for g in grids}
    lines: list[str] = ["# Cell occupancy (Smart-OCR v2)", ""]

    for page_no in sorted(by_page.keys()):
        grid = by_page[page_no]
        rows, cols = grid.rows, grid.cols
        grid_char: list[list[str]] = [["." for _ in range(cols)] for _ in range(rows)]

        for bid, blk in flat_blocks.items():
            bid_str = str(blk.get("block_id") or bid)
            m = _BLOCK_PAGE_RE.match(bid_str)
            if not m or int(m.group(1)) != page_no:
                continue
            kind = str(blk.get("kind") or "")
            ch = {
                "printed_question": "Q",
                "handwritten_answer": "A",
                "marking_box": "M",
                "header_footer": "H",
                "instructions": "I",
            }.get(kind, "#")
            cells = blk.get("cells")
            if not isinstance(cells, list):
                continue
            for rc in cells_from_range_strings(grid, [str(x) for x in cells]):
                r, c = rc[0] - 1, rc[1] - 1
                if 0 <= r < rows and 0 <= c < cols:
                    if grid_char[r][c] == ".":
                        grid_char[r][c] = ch
                    elif ch == "A" and grid_char[r][c] == "Q":
                        grid_char[r][c] = "A"

        for item in items:
            anns = item.get("annotations") or []
            if not isinstance(anns, list):
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                try:
                    ap = int(ann.get("page", 1))
                except (TypeError, ValueError):
                    ap = 1
                if ap != page_no:
                    continue
                cr = ann.get("comment_rows")
                if not isinstance(cr, list):
                    continue
                for rstr in cr:
                    for rc in cells_from_range_strings(grid, [str(rstr)]):
                        r, c = rc[0] - 1, rc[1] - 1
                        if 0 <= r < rows and 0 <= c < cols:
                            grid_char[r][c] = "*"

        lines.append(f"## Page {page_no}")
        lines.append("")
        lines.append("```")
        lines.append(_header(cols))
        for r in range(rows):
            lines.append(f"{r+1:2d} " + "".join(grid_char[r][c].ljust(2) for c in range(cols)))
        lines.append("```")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
