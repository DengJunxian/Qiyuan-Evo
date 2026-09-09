"""Content identity of the code and configuration used in competition runs."""
from hashlib import sha256
from pathlib import Path
import platform

from .models import digest


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    files = [p for folder in ("agents", "core", "engine", "policy") for p in (root / folder).rglob("*.py")]
    files += [root / p for p in ("config.py", "config.yaml", "simulation_runner.py", "simulation_ipc.py", "requirements-lock.txt")]
    hashes = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()}
    return {"source_hash": digest(hashes), "python": platform.python_version(), "files": hashes}
