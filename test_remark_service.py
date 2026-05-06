import json
import numpy as np
from pathlib import Path

import pytest

from remark_service import (
    BASE_PDF_RELATIVE_PATH,
    FreeSpaceBBox,
    InvalidRemarksFormat,
    PageOutOfRange,
    PageFreeSpaceResult,
    Remark,
    RemarkNotFound,
    RemarkService,
    SessionNotFound,
    SessionStore,
    _percent_to_pdf_bbox,
    _positions_from_bbox_pages,
    _zone_to_bbox,
    _parse_remarks,
    load_base_pdf_bytes,
    # New algorithm functions
    _otsu_binarise,
    _ink_proximity_map,
    _sample_proximity_grid,
    _greedy_diverse_select,
    _candidates_to_free_zones,
)


def _minimal_pdf(width: float = 595.0, height: float = 842.0) -> bytes:
    offsets: list[int] = []
    body = b""

    def obj(num: int, content: bytes) -> bytes:
        offsets.append(len(body))
        return f"{num} 0 obj\n".encode() + content + b"\nendobj\n"

    o1 = obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    o2 = obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    o3 = obj(3, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] >>".encode())

    header = b"%PDF-1.4\n"
    body_bytes = header + o1 + o2 + o3
    xref_offset = len(body_bytes)
    xref = (
        "xref\n0 4\n0000000000 65535 f \n"
        + "".join(f"{len(header) + off:010d} 00000 n \n" for off in offsets)
    ).encode()
    trailer = f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return body_bytes + xref + trailer


SAMPLE_REMARKS_LIST = [
    {"key": "Q1", "text": "Excellent reasoning"},
    {"key": "Q2", "text": "Needs more detail"},
    {"key": "Q3", "text": "Calculation error on line 4"},
]

SAMPLE_REMARKS_DICT = {"Q1": "Excellent reasoning", "Q2": "Needs more detail"}
BASE_PDF_TEST_RELATIVE_PATH = BASE_PDF_RELATIVE_PATH


@pytest.fixture()
def pdf_bytes() -> bytes:
    return _minimal_pdf()


@pytest.fixture()
def store() -> SessionStore:
    return SessionStore()


@pytest.fixture()
def svc() -> RemarkService:
    return RemarkService()


@pytest.fixture()
def session_id(store, pdf_bytes) -> str:
    return store.create(pdf_bytes, SAMPLE_REMARKS_LIST, filename="test.pdf", dpi=72)


@pytest.fixture()
def session_id_no_remarks(store, pdf_bytes) -> str:
    return store.create(pdf_bytes, None, filename="blank.pdf")


class TestParseRemarks:
    def test_list_of_dicts(self):
        result = _parse_remarks(SAMPLE_REMARKS_LIST)
        assert len(result) == 3
        assert all(isinstance(r, Remark) for r in result)

    def test_dict_shorthand(self):
        result = _parse_remarks(SAMPLE_REMARKS_DICT)
        assert {r.key for r in result} == {"Q1", "Q2"}

    def test_dict_rich_values(self):
        raw = {"Q1": {"text": "Good", "category": "positive", "color": "#4CAF50"}}
        result = _parse_remarks(raw)
        assert result[0].category == "positive"
        assert result[0].color == "#4CAF50"

    def test_list_of_strings(self):
        result = _parse_remarks(["First remark", "Second remark"])
        assert result[0].key == "1"
        assert result[0].text == "First remark"

    def test_json_string_input(self):
        result = _parse_remarks(json.dumps(SAMPLE_REMARKS_LIST))
        assert len(result) == 3

    def test_empty_list(self):
        assert _parse_remarks([]) == []

    def test_empty_dict(self):
        assert _parse_remarks({}) == []

    def test_invalid_type_raises(self):
        with pytest.raises(InvalidRemarksFormat):
            _parse_remarks(42)


class TestSessionStore:
    def test_create_returns_uuid_string(self, store, pdf_bytes):
        sid = store.create(pdf_bytes, SAMPLE_REMARKS_LIST)
        assert isinstance(sid, str) and len(sid) == 36

    def test_get_missing_raises(self, store):
        with pytest.raises(SessionNotFound):
            store.get("ghost")

    def test_dpi_clamped(self, store, pdf_bytes):
        sid_low = store.create(pdf_bytes, None, dpi=10)
        sid_high = store.create(pdf_bytes, None, dpi=9999)
        assert store.get(sid_low).dpi == 72
        assert store.get(sid_high).dpi == 300


class TestHelpers:
    def test_percent_to_pdf_bbox_orders_values(self):
        x1, x2, y1, y2 = _percent_to_pdf_bbox(10, 40, 20, 30, 600.0, 800.0)
        assert x1 < x2
        assert y1 < y2
        assert x1 == pytest.approx(155.4)
        assert x2 == pytest.approx(297.6)

    def test_zone_to_bbox_builds_contract(self):
        class DummyZone:
            x_start_percent = 20.0
            x_end_percent = 30.0
            y_start_percent = 40.0
            y_end_percent = 50.0
            score = 0.85

        box = _zone_to_bbox(DummyZone(), page=2, page_width=600.0, page_height=800.0)
        assert isinstance(box, FreeSpaceBBox)
        assert box.page == 2
        assert box.score == pytest.approx(0.85)
        assert box.pdf_x1 < box.pdf_x2
        assert box.pdf_y1 < box.pdf_y2

    def test_positions_from_bbox_pages(self):
        pages = [
            PageFreeSpaceResult(
                page=1,
                bboxes=[
                    FreeSpaceBBox(
                        page=1,
                        score=0.9,
                        x_start_percent=10,
                        x_end_percent=20,
                        y_start_percent=30,
                        y_end_percent=40,
                        pdf_x1=10,
                        pdf_y1=10,
                        pdf_x2=20,
                        pdf_y2=20,
                    )
                ],
            )
        ]
        positions = _positions_from_bbox_pages(pages, page=1, n=1, cw=1000, ch=1000)
        assert positions == [(298, 351)]

    def test_load_base_pdf_bytes_uses_relative_path(self, tmp_path):
        test_pdf = tmp_path / "tmp.pdf"
        test_pdf.write_bytes(_minimal_pdf())
        old = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)
            raw = load_base_pdf_bytes("tmp.pdf")
            assert raw[:5] == b"%PDF-"
        finally:
            os.chdir(old)


class TestBBoxDetection:
    def test_detect_free_space_bboxes_returns_pages(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=5)
        assert len(pages) == 1
        assert pages[0].page == 1
        assert isinstance(pages[0].bboxes, list)

    def test_bbox_contract_fields_present(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=5)
        if not pages[0].bboxes:
            pytest.skip("No bboxes detected on minimal synthetic pdf.")
        item = pages[0].bboxes[0].to_dict()
        for key in (
            "page",
            "score",
            "x_start_percent",
            "x_end_percent",
            "y_start_percent",
            "y_end_percent",
            "pdf_x1",
            "pdf_y1",
            "pdf_x2",
            "pdf_y2",
        ):
            assert key in item

    def test_percent_bbox_ranges(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=10, min_score=0.0)
        for p in pages:
            for b in p.bboxes:
                assert 0.0 <= b.x_start_percent <= b.x_end_percent <= 100.0
                assert 0.0 <= b.y_start_percent <= b.y_end_percent <= 100.0

    def test_pdf_bbox_ranges(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=10, min_score=0.0)
        assert pages and len(pages) == 1
        for b in pages[0].bboxes:
            assert 0.0 <= b.pdf_x1 <= b.pdf_x2 <= 595.0
            assert 0.0 <= b.pdf_y1 <= b.pdf_y2 <= 842.0


class TestPlacementCompatibility:
    def test_place_unknown_key_raises(self, svc, store, session_id):
        with pytest.raises(RemarkNotFound):
            svc.place(store, session_id, page=1, key="BAD", x=0, y=0, canvas_width=100, canvas_height=100)

    def test_page_zero_raises(self, svc, store, session_id):
        with pytest.raises(PageOutOfRange):
            svc.place(store, session_id, page=0, key="Q1", x=0, y=0, canvas_width=100, canvas_height=100)

    def test_auto_place_free_space_fallback_when_base_pdf_missing(self, svc, store, session_id):
        placed = svc.auto_place(
            store,
            session_id,
            strategy="free_space",
            use_base_pdf_for_free_space=True,
            base_pdf_relative_path="missing.pdf",
        )
        assert len(placed) == 3


@pytest.mark.skipif(
    not (Path.cwd() / BASE_PDF_TEST_RELATIVE_PATH).exists(),
    reason="Configured base PDF path does not exist in current workspace.",
)
class TestBasePdfIntegration:
    def test_base_pdf_path_loads(self):
        raw = load_base_pdf_bytes(BASE_PDF_TEST_RELATIVE_PATH)
        assert raw[:5] == b"%PDF-"

    def test_detect_free_space_bboxes_whole_pdf(self, svc):
        raw = load_base_pdf_bytes(BASE_PDF_TEST_RELATIVE_PATH)
        pages = svc.detect_free_space_bboxes(raw)
        assert len(pages) >= 1
        total = sum(len(p.bboxes) for p in pages)
        assert total >= 1


# ─── New algorithm unit tests ─────────────────────────────────────────────────


class TestOtsuBinarise:
    def test_all_white_image(self):
        img = np.full((100, 100), 255, dtype=np.uint8)
        result = _otsu_binarise(img)
        assert result.all()   # all pixels > threshold → all background=True

    def test_half_black_half_white(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255   # right half white
        result = _otsu_binarise(img)
        assert not result[:, :50].any()   # left half = ink = False
        assert result[:, 50:].all()       # right half = background = True

    def test_returns_bool_array(self):
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = _otsu_binarise(img)
        assert result.dtype == bool
        assert result.shape == img.shape


class TestInkProximityMap:
    def test_isolated_ink_pixel_creates_halo(self):
        # A 10×10 block of ink so the Gaussian peak is detectable.
        # Single-pixel ink at sigma=10 gives peak ≈ 1/(2π×100) ≈ 0.0016 —
        # much lower than 0.1.  A 10×10 block raises peak to ~0.016×100 ≈ 0.16.
        ink = np.zeros((100, 100), dtype=bool)
        ink[45:55, 45:55] = True   # 10×10 ink block at centre
        prox = _ink_proximity_map(ink, sigma_px=5.0)
        assert prox[50, 50] > 0.1
        assert prox[0, 0] < prox[50, 50]   # corner is farther from ink

    def test_all_background_gives_near_zero(self):
        ink = np.zeros((50, 50), dtype=bool)
        prox = _ink_proximity_map(ink, sigma_px=10.0)
        assert prox.max() < 1e-10

    def test_all_ink_gives_near_one(self):
        ink = np.ones((50, 50), dtype=bool)
        prox = _ink_proximity_map(ink, sigma_px=5.0)
        assert prox[25, 25] > 0.9   # interior surrounded by ink


class TestSampleProximityGrid:
    def test_returns_correct_count(self):
        prox = np.zeros((100, 200), dtype=np.float32)
        candidates = _sample_proximity_grid(prox, sample_rows=5, sample_cols=10)
        assert len(candidates) == 50

    def test_all_values_in_range(self):
        prox = np.random.rand(200, 300).astype(np.float32)
        candidates = _sample_proximity_grid(prox, 10, 10)
        for val, y_pct, x_pct in candidates:
            assert 0.0 <= y_pct <= 100.0
            assert 0.0 <= x_pct <= 100.0
            assert 0.0 <= val

    def test_zero_proximity_map_all_zero(self):
        prox = np.zeros((100, 100), dtype=np.float32)
        candidates = _sample_proximity_grid(prox, 5, 5)
        for val, _, _ in candidates:
            assert val == pytest.approx(0.0)


class TestGreedyDiverseSelect:
    def test_respects_top_n(self):
        candidates = [(0.01 * i, float(i * 20), float(i * 20)) for i in range(10)]
        selected = _greedy_diverse_select(candidates, top_n=3, min_spacing_pct=5.0, max_proximity=0.5)
        assert len(selected) <= 3

    def test_enforces_min_spacing(self):
        candidates = [
            (0.01, 50.0, 50.0),
            (0.02, 51.0, 51.0),   # too close to first (Euclidean < 10)
            (0.03, 10.0, 10.0),   # far enough
        ]
        selected = _greedy_diverse_select(candidates, top_n=5, min_spacing_pct=10.0, max_proximity=1.0)
        ys = [y for _, y, _ in selected]
        xs = [x for _, _, x in selected]
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                dist = ((ys[i] - ys[j]) ** 2 + (xs[i] - xs[j]) ** 2) ** 0.5
                assert dist >= 10.0, f"Points {i} and {j} too close: {dist:.2f}"

    def test_rejects_above_max_proximity(self):
        candidates = [
            (0.10, 10.0, 10.0),
            (0.50, 90.0, 90.0),
        ]
        selected = _greedy_diverse_select(candidates, top_n=5, min_spacing_pct=5.0, max_proximity=0.05)
        assert len(selected) == 0   # both above threshold

    def test_empty_input(self):
        selected = _greedy_diverse_select([], top_n=5, min_spacing_pct=5.0, max_proximity=0.1)
        assert selected == []


class TestCandidatesToFreeZones:
    def test_score_inverts_proximity(self):
        selected = [(0.02, 50.0, 50.0)]
        zones = _candidates_to_free_zones(selected, sample_rows=10, sample_cols=10)
        assert zones[0].score == pytest.approx(1.0 - 0.02, abs=0.001)

    def test_bbox_centred_on_anchor(self):
        selected = [(0.0, 50.0, 50.0)]
        zones = _candidates_to_free_zones(selected, sample_rows=10, sample_cols=10)
        z = zones[0]
        # cell is 10% wide (100/10) and 10% tall; centred at 50%
        assert z.x_start_percent == pytest.approx(45.0)
        assert z.x_end_percent   == pytest.approx(55.0)
        assert z.y_start_percent == pytest.approx(45.0)
        assert z.y_end_percent   == pytest.approx(55.0)

    def test_bbox_clamped_at_edges(self):
        selected = [(0.0, 1.5, 1.5)]
        zones = _candidates_to_free_zones(selected, sample_rows=10, sample_cols=10)
        z = zones[0]
        assert z.x_start_percent >= 0.0
        assert z.y_start_percent >= 0.0


class TestBBoxDetectionNewAlgorithm:
    def test_white_page_returns_zones(self, svc, pdf_bytes):
        # Blank white page → all proximity=0 → all candidates pass threshold → zones found
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=5)
        assert len(pages) == 1
        assert len(pages[0].bboxes) > 0

    def test_score_range(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=10, proximity_thresh=1.0)
        for p in pages:
            for b in p.bboxes:
                assert 0.0 <= b.score <= 1.0

    def test_pdf_coords_within_page(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(pdf_bytes, top_n=10, proximity_thresh=1.0)
        for p in pages:
            for b in p.bboxes:
                assert 0.0 <= b.pdf_x1 <= b.pdf_x2 <= 595.0
                assert 0.0 <= b.pdf_y1 <= b.pdf_y2 <= 842.0

    def test_diversity_spacing(self, svc, pdf_bytes):
        pages = svc.detect_free_space_bboxes(
            pdf_bytes, top_n=5, proximity_thresh=1.0, min_spacing_pct=10.0
        )
        bboxes = pages[0].bboxes
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                cy_i = (bboxes[i].y_start_percent + bboxes[i].y_end_percent) / 2
                cx_i = (bboxes[i].x_start_percent + bboxes[i].x_end_percent) / 2
                cy_j = (bboxes[j].y_start_percent + bboxes[j].y_end_percent) / 2
                cx_j = (bboxes[j].x_start_percent + bboxes[j].x_end_percent) / 2
                dist = ((cy_i - cy_j) ** 2 + (cx_i - cx_j) ** 2) ** 0.5
                assert dist >= 9.5, f"Zones {i} and {j} too close: {dist:.2f}%"


# ─── Real PDF integration tests ───────────────────────────────────────────────

_REAL_PDFS = ["test_original_pdf1.pdf", "test_original_pdf2.pdf"]
_ANY_REAL_PDF_EXISTS = any((Path.cwd() / f).exists() for f in _REAL_PDFS)


@pytest.mark.skipif(
    not _ANY_REAL_PDF_EXISTS,
    reason="Neither test_original_pdf1.pdf nor test_original_pdf2.pdf found in cwd.",
)
class TestRealPdfIntegration:
    @pytest.fixture(params=_REAL_PDFS)
    def real_pdf_bytes(self, request):
        path = Path.cwd() / request.param
        if not path.exists():
            pytest.skip(f"{request.param} not found")
        return path.read_bytes()

    def test_free_zones_found(self, svc, real_pdf_bytes):
        pages = svc.detect_free_space_bboxes(real_pdf_bytes, top_n=10)
        total = sum(len(p.bboxes) for p in pages)
        assert total >= 1, "Expected at least one free zone on a real handwritten PDF"

    def test_scores_valid(self, svc, real_pdf_bytes):
        pages = svc.detect_free_space_bboxes(real_pdf_bytes, top_n=10)
        for p in pages:
            for b in p.bboxes:
                assert 0.0 <= b.score <= 1.0

    def test_pdf_coords_in_page_bounds(self, svc, real_pdf_bytes):
        pages = svc.detect_free_space_bboxes(real_pdf_bytes, top_n=10)
        for p in pages:
            for b in p.bboxes:
                assert b.pdf_x1 >= 0.0
                assert b.pdf_y1 >= 0.0
                assert b.pdf_x1 <= b.pdf_x2
                assert b.pdf_y1 <= b.pdf_y2

    def test_zones_spatially_diverse(self, svc, real_pdf_bytes):
        pages = svc.detect_free_space_bboxes(real_pdf_bytes, top_n=5, min_spacing_pct=12.0)
        for p in pages:
            bboxes = p.bboxes
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    cy_i = (bboxes[i].y_start_percent + bboxes[i].y_end_percent) / 2
                    cx_i = (bboxes[i].x_start_percent + bboxes[i].x_end_percent) / 2
                    cy_j = (bboxes[j].y_start_percent + bboxes[j].y_end_percent) / 2
                    cx_j = (bboxes[j].x_start_percent + bboxes[j].x_end_percent) / 2
                    dist = ((cy_i - cy_j) ** 2 + (cx_i - cx_j) ** 2) ** 0.5
                    assert dist >= 9.5, (
                        f"Page {p.page} zones {i} and {j} too close: {dist:.2f}%"
                    )

    def test_high_score_zones_exist(self, svc, real_pdf_bytes):
        pages = svc.detect_free_space_bboxes(real_pdf_bytes, top_n=10)
        for p in pages:
            if p.bboxes:
                best_score = max(b.score for b in p.bboxes)
                assert best_score >= 0.90, (
                    f"Page {p.page}: best zone score {best_score:.4f} is unexpectedly low"
                )
