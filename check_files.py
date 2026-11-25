"""Verify every archived payload file using the supplied SHA256 manifest."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for index,(relative,expected) in enumerate(manifest.items(),1):
        path=root/relative
        if not path.is_file():raise FileNotFoundError(path)
        if path.stat().st_size!=expected['bytes']:raise ValueError(f'Size mismatch: {relative}')
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected['sha256']:raise ValueError(f'SHA256 mismatch: {relative}')
        if index%1000==0:print(f'Checked {index}/{len(manifest)} files',flush=True)
    print(f'PASS: {len(manifest)} payload files match SHA256.',flush=True)


if __name__=='__main__':main()
