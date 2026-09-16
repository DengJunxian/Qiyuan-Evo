"""Verify illustrative provenance, internally consistent numbers and route isolation."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from core.competition_cockpit import ExperimentArchive
from ui.demo_gallery import gallery_records
from ui.evolution_case import CASE_ID, CHAPTERS, CHECKS, MISSING, PRICES, quality_score, max_drawdown

ROOT = Path(__file__).resolve().parents[1]


def test_demo_screenshots_match_captured_chapters_and_never_claim_measured_runs():
    directory = ROOT / 'static/evolution_story_gallery'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['data_kind'] == 'synthetic_demo'
    records = gallery_records()
    assert len(records) == len(CHAPTERS) == 8
    for i, row in enumerate(records, 1):
        assert row['case_id'] == CASE_ID and row['data_kind'] == 'synthetic_demo'
        assert row['chapter'] == i and row['file'] == CHAPTERS[i-1][3]
        assert not {'experiment_id', 'baseline_run_id', 'optimizer_backend'} & row.keys()
        query = parse_qs(urlparse(row['capture_url']).query)
        assert query['page'] == ['演进案例'] and query['chapter'] == [str(i)]
        path = directory / row['file']
        assert sha256(path.read_bytes()).hexdigest() == row['sha256']
        with Image.open(path) as image:
            assert image.size == (1600, 900)


def test_review_totals_and_drawdowns_agree_with_the_story():
    assert quality_score('baseline') == 62
    assert quality_score('evolved') == 92
    assert len(CHECKS) == 10 and len(CHECKS) - len(MISSING) == 6
    drawdowns = [max_drawdown(prices) * 100 for prices in PRICES.values()]
    assert drawdowns == pytest.approx([6.8, 5.9, 4.3])
    assert drawdowns[0] - drawdowns[2] == pytest.approx(2.5)
    assert all(len(series) == 11 and series[0] == 100 for series in PRICES.values())


@pytest.mark.parametrize('chapter,selected', [('6',6), ('8',8), ('invalid',1), ('99',8), ('0',1)])
def test_case_route_works_without_archive_and_validates_chapter(monkeypatch, chapter, selected):
    def unavailable(*args, **kwargs):
        raise AssertionError('Demonstration must not initialize experiment storage')
    monkeypatch.setattr(ExperimentArchive, '__init__', unavailable)
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60)
    app.query_params.update({'page':'演进案例', 'chapter':chapter, 'appearance':'dark'})
    app.run()
    assert not app.exception
    html = ' '.join(item.value for item in app.get('html'))
    assert f'data-selected="{selected}"' in html
    assert '设计案例 · 示例数据' in html
    assert 'DEMO-LIQ-01 · 流程演示，非实测结果' in html
    assert '计划已更新 · 待开始执行' in html
