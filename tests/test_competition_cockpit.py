from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from core.competition_cockpit import (DemoJobs, ExperimentArchive, PROFILES, capabilities, demo_task,
                                      generations, validate_comparison)
from core.competition_service import CompetitionService
from core.team_evolution.config import baseline_config
from core.team_evolution.models import digest
from ui.components.evolution import agent_states, events_from_trace, graph_svg, metric_delta


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    observations=[]
    root=tmp_path_factory.mktemp("observed-cockpit")
    service=CompetitionService(root,backend="local",observer=observations.append)
    task=demo_task("historical_analysis")
    result=service.run_before_after(task)
    return service,result,observations


def test_profiles_are_explicit_and_scale_keeps_fixed_input():
    small,full=demo_task(),demo_task(scale="完整实验")
    assert small.seed==full.seed==42
    assert small.inputs==full.inputs and small.evaluation_rules==full.evaluation_rules
    assert small.constraints['simulation_days']<full.constraints['simulation_days']
    assert PROFILES['OFFLINE FALLBACK'].backend=='local'
    assert not PROFILES['REPRODUCIBLE DEMO'].cloud_optimizer
    assert PROFILES['ONLINE LIVE'].cloud_optimizer
    assert capabilities()['role_count']==len(baseline_config().roles)+1
    assert capabilities()['task_count']==3


def test_observer_contains_actual_running_finished_feedback_and_promotion(observed):
    _,result,observations=observed
    kinds={o['event'] for o in observations}
    assert {'run_started','agent_started','tool_finished','agent_finished','baseline_evaluated',
            'evolution_started','candidate_proposed','evolution_decided','comparison_saved'}<=kinds
    start=next(o for o in observations if o['event']=='agent_started')
    assert start['trace']['agent_events'][0]['status']=='running'
    assert result['baseline']['agent_events'][0]['status']=='completed'
    assert len(next(o for o in observations if o['event']=='candidate_proposed')['history']['actions'])>=2
    assert result['promotion']['accepted']
    assert all('input_summary' in e and 'evidence_ids' in e for e in result['evolved']['agent_events'])
    produced={e['agent']:e['evidence_ids'] for e in result['evolved']['agent_events']}
    assert produced['historian']==['historical_comparison']
    assert produced['quant']==['quant']
    assert all(c['started_at']<=c['finished_at'] for c in result['evolved']['tool_calls'])
    assert json.loads(json.dumps(observations,allow_nan=False))==observations


def test_broken_or_mutating_observer_cannot_change_financial_results(observed,tmp_path):
    _,result,_=observed
    def broken(value):
        if value.get('trace'):
            value['trace']['artifacts'].clear()
            value['trace']['team_config']['topology']['active_agents'].clear()
        raise RuntimeError('display outage')
    replay=CompetitionService(tmp_path,backend='local',observer=broken).run_baseline(result['baseline']['task'])
    assert replay['artifacts']==result['baseline']['artifacts']
    assert replay['semantic_hash']==result['baseline']['semantic_hash']


def test_invalid_or_mismatched_artifact_is_not_presented(observed,tmp_path):
    _,result,_=observed
    archive=ExperimentArchive(tmp_path)
    assert archive.comparisons()==[] and not (tmp_path/'experience.sqlite3').exists()
    (tmp_path/'before_after.json').write_text('{broken')
    assert archive.comparisons()==[] and archive.issues
    changed=deepcopy(result)
    changed['evolved']['task']['seed']=99
    (tmp_path/'before_after.json').write_text(json.dumps(changed))
    assert ExperimentArchive(tmp_path).comparisons()==[]
    with pytest.raises(ValueError): validate_comparison(changed)
    changed=deepcopy(result)
    changed['evolved']['runtime']['dataset_hash']='mismatch'
    with pytest.raises(ValueError): validate_comparison(changed)
    changed=deepcopy(result)
    changed['baseline']['evaluation']['quality_score']=1.0
    with pytest.raises(ValueError): validate_comparison(changed)


def test_archive_reading_preserves_manifest_and_counts_real_memory_references(observed):
    service,result,_=observed
    root=service.store.root
    before=(root/'source_manifest.json').read_bytes()
    archive=ExperimentArchive(root)
    experiences=archive.experiences()
    refs=result['evolved']['plan']['retrieved_experience_ids']
    assert refs
    for eid in refs:
        row=next(x for x in experiences if x['experience_id']==eid)
        assert row['times_reused']>=1 and result['evolved']['run_id'] in row['used_by']
    assert before==(root/'source_manifest.json').read_bytes()


def test_generations_only_show_actual_configs_for_identical_tasks(observed):
    _,result,_=observed
    rows=generations([result,result],result['baseline']['task'])
    assert len(rows)==2
    assert [r['generation'] for r in rows]==[0,1]
    assert [len(r['config']['topology']['active_agents']) for r in rows]==[5,7]
    different=deepcopy(result['baseline']['task']);different['seed']=99
    assert generations([result],different)==[]


def test_similar_task_reads_successful_configuration_and_records_experience(observed):
    service, result, _ = observed
    similar = deepcopy(result["baseline"]["task"])
    similar["task_id"] = "historical_analysis_followup"
    similar["objective"] += "补充说明偏差来源。"
    trace = service.run_evolved(similar)
    assert trace["plan"]["retrieved_experience_ids"]
    assert trace["team_config"]["version"] == result["evolved"]["team_config"]["version"]
    assert trace["team_config"]["tool_policy"]["executor"]["preferred"] == "historical_replay"
    assert trace["evaluation"]["task_success"]


def test_archive_retains_earlier_comparisons_when_latest_index_is_replaced(observed):
    service,result,_=observed
    second=service.run_before_after(result['baseline']['task'])
    (service.store.root/'before_after.json').write_text(json.dumps(second))
    all_comparisons=ExperimentArchive(service.store.root).comparisons()
    assert len(all_comparisons)==2
    assert all_comparisons[0]['baseline']['run_id']==result['baseline']['run_id']
    assert all_comparisons[1]['baseline']['run_id']==second['baseline']['run_id']


def test_graph_states_events_and_html_escape(observed):
    _,result,observations=observed
    trace=result['evolved']
    assert set(agent_states(trace).values())=={'COMPLETED'}
    running=next(o['trace'] for o in observations if o['event']=='agent_started')
    assert agent_states(running)['planner']=='PLANNING'
    assert agent_states(running)['reporter']=='WAITING'
    raw=graph_svg(trace)
    assert '历史分析员' in raw and '量化分析员' in raw and '并行' in raw
    assert '<svg' in raw and '<script' not in raw
    malicious=deepcopy(trace);malicious['task_id']='</title><script>alert(1)</script>'
    assert '<script' not in graph_svg(malicious)
    events=events_from_trace(result['baseline'])
    feedback=[e for e in events if e['kind']=='feedback']
    assert [e['summary'] for e in feedback]==[c['problem'] for c in result['baseline']['critiques']]
    assert all(e['run_id']==result['baseline']['run_id'] for e in events)


def test_metric_arrows_preserve_regressions_and_unmeasured_values():
    assert metric_delta(100,82,lower=True)==('↓ 18.0%','good')
    assert metric_delta(10,12,lower=True)==('↑ 20.0%','warn')
    assert metric_delta(.61,.89,ratio=True)==('↑ 28.0 个百分点','good')
    assert metric_delta(0,0)==('— 持平','neutral')
    assert metric_delta(None,0)==('— 未测量','neutral')


def test_background_startup_failure_is_saved_and_does_not_stick(monkeypatch,tmp_path):
    def unavailable(*args,**kwargs): raise ConnectionError('test backend outage')
    monkeypatch.setattr(CompetitionService,'__init__',unavailable)
    jobs=DemoJobs(tmp_path)
    first=jobs.start(profile='OFFLINE FALLBACK')
    deadline=time.monotonic()+5
    while jobs.snapshot(first)['status'] not in {'COMPLETED','FAILED'} and time.monotonic()<deadline: time.sleep(.01)
    state=jobs.snapshot(first)
    assert state['status']=='FAILED' and state['error']['type']=='ConnectionError'
    assert (Path(state['output_dir'])/'demo_failure.json').exists()
    second=jobs.start(profile='OFFLINE FALLBACK')
    assert second!=first
    deadline=time.monotonic()+5
    while jobs.snapshot(second)['status'] not in {'COMPLETED','FAILED'} and time.monotonic()<deadline: time.sleep(.01)
    assert jobs.snapshot(second)['status']=='FAILED'
