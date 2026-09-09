"""Public releases must rely on local environment credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_model_credentials_are_empty_without_environment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("DEEPSEEK_API_KEY", "ZHIPUAI_API_KEY", "ZHIPU_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(root)
    code = """
import json
from config import SimConfig, DEFAULT_ZHIPU_API_KEY, DEFAULT_DEEPSEEK_API_KEY
settings = SimConfig()
print(json.dumps([bool(DEFAULT_ZHIPU_API_KEY), bool(DEFAULT_DEEPSEEK_API_KEY),
                 bool(settings.ZHIPU_API_KEY), bool(settings.DEEPSEEK_API_KEY)]))
"""
    result = subprocess.check_output([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True)
    assert json.loads(result) == [False, False, False, False]
