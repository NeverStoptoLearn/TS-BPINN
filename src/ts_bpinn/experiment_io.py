"""Run records shared in form, but independently packaged by each project."""
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import json
import platform
import sys

def start_run(config, output=None):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(output) if output else Path('runs') / stamp
    path.mkdir(parents=True, exist_ok=False)
    (path / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    versions = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}
    metadata = {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
                'created_utc': stamp, 'validation_scope': 'example run; not a paper-wide replication'}
    (path / 'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return path

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
