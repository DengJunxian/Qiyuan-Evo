from pathlib import Path
from copy import deepcopy
from hashlib import sha1
import json

from streamlit.testing.v1 import AppTest

from core.competition_cockpit import ExperimentArchive
from ui.demo_gallery import gallery_records, domain_gallery_records
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
    html = ' '.join(item.value for item in app.get('html'))
    assert html.count('<figure class="qg-image">') == 16
    assert '已完成实验截图' in html and '原 Civitas 项目截图' in html
    assert not app.exception
    case=deepcopy(ExperimentArchive().comparisons()[0])
    assert 'DeepSeek' in runtime_record(case)[1]
    case['evolution_history'][0]['optimizer']['backend']='LOCAL FALLBACK'
    assert 'DeepSeek' not in runtime_record(case)[1]
    before=case['baseline'];after=case['evolved']
    assert '新增' in graph_svg(after,previous=before)
    assert '新增' not in graph_svg(before,previous=before)


def test_original_screenshots_match_source_git_blobs_and_keep_their_origin():
    manifest=json.loads((ROOT/'static/showcase_gallery/manifest.json').read_text())
    assert manifest['source_repository']=='https://github.com/DengJunxian/Civitas-Economica-Demo'
    rows=domain_gallery_records()
    assert len(rows)==8
    for row in rows:
        content=(ROOT/'static/showcase_gallery'/row['file']).read_bytes()
        assert sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()==row['git_blob_sha']
        assert 'experiment_id' not in row and 'baseline_run_id' not in row


def test_home_gallery_is_available_without_experiment_store_or_runtime(monkeypatch):
    def unavailable(*args, **kwargs): raise ConnectionError('offline')
    monkeypatch.setattr(ExperimentArchive,'__init__',unavailable)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert app.session_state['entry']=='展示页面'
    html=' '.join(item.value for item in app.get('html'))
    assert html.count('<figure class="qg-image">')==16
    assert '进入实验' in html and 'app/static/showcase_gallery/01-policy-input.png' in html
    assert not app.selectbox and not app.caption


def test_one_corrupt_collection_does_not_hide_the_other(tmp_path,monkeypatch):
    from ui import demo_gallery
    (tmp_path/'manifest.json').write_text('{bad json')
    monkeypatch.setattr(demo_gallery,'GALLERY',tmp_path)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception and app.warning
    html=' '.join(item.value for item in app.get('html'))
    assert html.count('<figure class="qg-image">')==8
    assert 'app/static/showcase_gallery/08-authenticity-overview.png' in html
