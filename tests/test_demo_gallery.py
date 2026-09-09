from pathlib import Path
from copy import deepcopy

from streamlit.testing.v1 import AppTest

from core.competition_cockpit import ExperimentArchive
from ui.demo_gallery import gallery_records
from ui.components.evolution import graph_svg
from ui.team_evolution import runtime_record

ROOT=Path(__file__).resolve().parents[1]


def test_screenshots_are_bound_to_current_measured_experiments():
    records=gallery_records()
    cases={c['experiment_id']:c for c in ExperimentArchive().comparisons()}
    assert len(records)==8
    for row in records:
        case=cases[row['experiment_id']]
        assert row['baseline_run_id']==case['baseline']['run_id']
        assert row['source_hash']==case['evolved']['runtime']['source_hash']
        assert row['model']==case['evolution_history'][0]['optimizer']['model']
        assert row['optimizer_backend']=='OPENJIUWEN'
        assert (ROOT/'static/competition_gallery'/row['file']).stat().st_size>10000


def test_demo_navigation_prompt_memory_gallery_and_fallback_label():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.button(key='top_entry_演进实验').click().run()
    for view in ('compare','prompt','memory'):
        app.radio(key='ev_evolution_view').set_value(view).run()
        assert not app.exception
    app.button(key='top_entry_展示页面').click().run()
    assert len(app.selectbox(key='gallery_item').options)==8
    app.selectbox(key='gallery_item').set_value(3).run()
    assert not app.exception
    case=deepcopy(ExperimentArchive().comparisons()[0])
    assert 'DeepSeek' in runtime_record(case)[1]
    case['evolution_history'][0]['optimizer']['backend']='LOCAL FALLBACK'
    assert 'DeepSeek' not in runtime_record(case)[1]
    before=case['baseline'];after=case['evolved']
    assert '新增' in graph_svg(after,previous=before)
    assert '新增' not in graph_svg(before,previous=before)
