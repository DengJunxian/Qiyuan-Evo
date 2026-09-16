"""The presentation must retain the measured points and fail closed on bad evidence."""
from hashlib import sha256
import json

import pytest

from ui.showcase_visuals import chart_data, presentation_cases, visual_story


def test_chart_points_keep_measured_paths_and_zero_outcomes():
    cases = {c['baseline']['task']['task_type']: c for c in presentation_cases()}
    assert len(cases) == 3
    policy = cases['policy_report']
    chart = chart_data(policy)
    assert chart['series'][0]['values'] == policy['evolved']['artifacts']['simulation']['data']['prices']
    assert chart['series'][1]['values'] == policy['evolved']['artifacts']['control']['data']['prices']
    assert chart['metric'] == '+0.00'
    history = cases['historical_analysis']['evolved']['artifacts']['historical_comparison']['data']
    chart = chart_data(cases['historical_analysis'])
    assert chart['labels'] == history['dates']
    assert chart['series'][0]['values'] == pytest.approx([100 * p / history['observed_prices'][0] for p in history['observed_prices']])
    assert '不代表样本外预测' in chart['note']
    regulatory = cases['regulatory_planning']['evolved']['artifacts']['counterfactual_search']['data']
    chart = chart_data(cases['regulatory_planning'])
    assert [s['values'] for s in chart['series']] == [p['prices'] for p in regulatory['plans']]
    assert chart['metric'] == '0.00' and '暂不干预' in chart['note']


@pytest.mark.parametrize('update_manifest', [False, True])
def test_presentation_rejects_hash_or_recomputed_score_mismatch(tmp_path, monkeypatch, update_manifest):
    case = presentation_cases()[0]
    original = json.dumps([case]).encode()
    case['evolved']['evaluation']['quality_score'] = 0.1234
    tampered = json.dumps([case]).encode()
    (tmp_path / 'before_after.json').write_bytes(tampered)
    (tmp_path / 'reference_manifest.json').write_text(json.dumps({'files': {
        'before_after.json': sha256(tampered if update_manifest else original).hexdigest()}}))
    monkeypatch.setenv('QIYUAN_REFERENCE_DIR', str(tmp_path))
    assert presentation_cases() == []
    assert visual_story(presentation_cases(), lambda row: '?page=test') == ('', '')
