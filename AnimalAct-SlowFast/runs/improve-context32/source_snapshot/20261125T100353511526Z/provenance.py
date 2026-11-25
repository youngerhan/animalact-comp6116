"""Record inputs, source fingerprints and installed versions for each new run."""
import hashlib
import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from clips import write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record_run(out, options):
    import torch
    project = Path(__file__).resolve().parent
    versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    inputs = {}
    for name in ('manifest', 'config', 'weights', 'initial'):
        path = getattr(options, name, None)
        if path is not None:
            inputs[name] = {'path': str(Path(path).resolve()), 'sha256': sha256(path)}
    source = {path.name: sha256(path) for path in sorted(project.glob('*.py'))}
    sources_dir = out / 'source_snapshot'
    sources_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot = sources_dir / stamp
    snapshot.mkdir()
    for path in project.glob('*.py'):
        (snapshot / path.name).write_bytes(path.read_bytes())
    record = dict(python=sys.version, platform=platform.platform(), packages=versions,
                  torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  command=sys.argv, inputs=inputs, source_sha256=source,
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
    write_json(out / f'provenance-{stamp}.json', record)
