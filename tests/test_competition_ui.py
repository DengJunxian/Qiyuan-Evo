from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.competition_service import CompetitionService
from core.team_evolution.tools import FinancialTools

ROOT = Path(__file__).resolve().parents[1]


def test_cockpit_and_seven_primary_pages_and_original_domain_pages():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert app.session_state['entry']=='自演进驾驶舱'
    assert app.button(key='ev_demo').label=='重新运行此案例'
    assert any('<svg' in item.proto.srcdoc for item in app.get('iframe'))
    for page in ('协作过程','演进实验','三类任务','金融政策风洞','历史验证','技术与复现'):
        app.button(key='top_entry_'+page).click().run()
        assert not app.exception,page
        if page=="历史验证":
            assert app.tabs[0].label=="已完成验证"
            assert app.get("plotly_chart")
    for page in ('成果展示','系统总览','政策实验','研判分析'):
        app.button(key='entry_'+page).click().run()
        assert not app.exception,page


def test_failed_tool_trace_does_not_crash_frontend(tmp_path,monkeypatch):
    def unavailable(*args,**kwargs): raise ConnectionError('test outage')
    monkeypatch.setattr(FinancialTools,'market_simulation',unavailable)
    trace=CompetitionService(tmp_path,backend='local').run_baseline('policy_report')
    assert not trace['evaluation']['task_success']
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.session_state['ev_result']=trace
    app.button(key='top_entry_协作过程').click().run()
    assert not app.exception and app.warning
    assert any('FAILED' in item.proto.srcdoc for item in app.get('iframe'))


def test_empty_and_corrupt_archive_has_no_invented_scores(tmp_path,monkeypatch):
    monkeypatch.setenv('QIYUAN_REFERENCE_DIR',str(tmp_path))
    (tmp_path/'before_after.json').write_text('{bad json')
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert any('暂无可复现实验结果' in x.value for x in app.info)
    assert any('IDLE' in item.proto.srcdoc for item in app.get('iframe'))
    for page in ('协作过程','演进实验','三类任务','技术与复现'):
        app.button(key='top_entry_'+page).click().run()
        assert not app.exception,page
