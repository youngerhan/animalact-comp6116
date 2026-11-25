"""Apply a SHA256-verified update to the previously extracted AnimalAct bundle.

Standard library only. The original files are backed up before replacement.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
LEGACY_RUNS = (
    "main-all-uniform224", "ablate-all-posweight", "ft256",
    "seed123-all-uniform224", "ablate-l34-sqrt2", "wholeframe224",
    "wf256", "pilot32-whole224",
)


def checked_path(root, relative):
    """Resolve one manifest member without allowing drive paths or traversal."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Manifest path must use forward slashes: {relative!r}")
    posix = PurePosixPath(relative)
    if (posix.is_absolute() or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    candidate = root.joinpath(*posix.parts)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Manifest path leaves its intended directory: {relative!r}")
    # Reject links even when they point inside the target: replacing a link can
    # surprise a user and following it can modify a different payload file.
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"Symbolic links are not update targets: {relative!r}")
        cursor = cursor.parent
    return candidate


def validated_entries(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty path-to-checksum mapping")
    seen = set()
    for relative, expected in value.items():
        checked_path(ROOT, relative)
        canonical = relative.casefold()
        if canonical in seen:
            raise ValueError(f"Case-colliding manifest path: {relative}")
        seen.add(canonical)
        if not isinstance(expected, dict):
            raise ValueError(f"Invalid checksum record: {relative}")
        size, digest = expected.get("bytes"), expected.get("sha256")
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"Invalid size or SHA256: {relative}")
    return value


def fingerprint(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def verify(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Size mismatch for {label}: {path}")
    if fingerprint(path) != expected:
        raise ValueError(f"SHA256 mismatch for {label}: {path}")


def verify_base(target, base_files):
    """Check required old checkpoints and the data/runtime base before writing."""
    required = ["runtime/python.exe", "AnimalAct/action_classes.json",
                "AnimalAct-SlowFast/data/main.json"]
    required += [f"AnimalAct-SlowFast/runs/{name}/best.pth" for name in LEGACY_RUNS]
    for relative in required:
        if relative not in base_files:
            raise ValueError(f"Update lacks a base checksum for {relative}")
        verify(checked_path(target, relative), base_files[relative], "base file")
    for prefix in ("AnimalAct/annotations/", "AnimalAct/splits/", "AnimalAct/videos/"):
        entries = [(relative, expected) for relative, expected in base_files.items()
                   if relative.startswith(prefix)]
        if not entries:
            raise ValueError(f"Update manifest lacks original base files under {prefix}")
        for relative, expected in entries:
            path = checked_path(target, relative)
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Incomplete original bundle; missing or changed file: {relative}")
    print("Base checkpoint, runtime and data checks passed.", flush=True)


def atomic_copy(source, destination):
    """Copy one known file using a temporary file in the destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=destination.name + ".update-",
                                         suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, handle, length=4 * 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            # This is only the exact temporary file created by this function.
            temporary.unlink(missing_ok=True)


def apply(target, root=ROOT, dry_run=False, verify_only=False):
    root = root.resolve()
    target = target.resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Extract the update outside the target bundle before applying it.")
    manifest = json.loads((root / "update-manifest.json").read_text(encoding="utf-8-sig"))
    version = manifest.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("Invalid update version")
    if manifest.get("base_version") != "v1.0.0-20261125":
        raise ValueError("This updater requires the documented v1.0.0-20261125 base.")
    files = validated_entries(manifest.get("files"), "files")
    base_files = validated_entries(manifest.get("base_files"), "base_files")
    payload = (root / "payload").resolve()
    if not payload.is_relative_to(root):
        raise ValueError("Payload directory leaves update root")
    if "SHA256SUMS.json" not in files:
        raise ValueError("Update must include its merged SHA256SUMS.json")
    print(f"Checking {len(files)} update files ({version})...", flush=True)
    for index, (relative, expected) in enumerate(files.items(), 1):
        verify(checked_path(payload, relative), expected, "update payload")
        if index % 100 == 0:
            print(f"Checked update payload: {index}/{len(files)}", flush=True)
    verify_base(target, base_files)
    if verify_only:
        for relative, expected in files.items():
            verify(checked_path(target, relative), expected, "installed update")
        print(f"PASS: all {len(files)} updated files are installed and match SHA256.", flush=True)
        return

    backup_root = checked_path(target, f"packaging/update-backups/{version}")
    plan = []
    for relative, expected in files.items():
        destination = checked_path(target, relative)
        if destination.exists():
            current = fingerprint(destination)
            if current == expected:
                continue  # Already installed; repeat application is safe.
            original = base_files.get(relative)
            if original is None or current != original:
                raise ValueError(f"Local file differs from both the original and update; preserved: {relative}")
            backup = checked_path(backup_root, relative)
            if backup.exists():
                verify(backup, current, "existing original-file backup")
        else:
            current, backup = None, None
        plan.append((relative, destination, current, backup))
    print(f"Will install {len(plan)} files; {len(files) - len(plan)} already match.", flush=True)
    print(f"Original-file backups: {backup_root}", flush=True)
    if dry_run:
        print("DRY RUN passed. No files were written.", flush=True)
        return
    for index, (relative, destination, current, backup) in enumerate(plan, 1):
        # Recheck immediately before touching a file in case it was edited while
        # a previous, large checkpoint was being copied.
        checked_path(target, relative)
        if current is not None:
            verify(destination, current, "unchanged local file")
            if not backup.exists():
                atomic_copy(destination, backup)
            verify(backup, current, "original-file backup")
        elif destination.exists():
            raise ValueError(f"Target appeared after preflight; preserved: {relative}")
        atomic_copy(checked_path(payload, relative), destination)
        verify(destination, files[relative], "installed update")
        if index % 100 == 0 or index == len(plan):
            print(f"Installed update: {index}/{len(plan)}", flush=True)
    print(f"PASS: {version} installed. Run VERIFY_IMPROVED.cmd next.", flush=True)
    print("Original-file backups were retained; no training was started.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Root of the extracted v1.0.0 full bundle")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and preview without writing files")
    mode.add_argument("--verify-only", action="store_true", help="Verify that every update file is installed")
    args = parser.parse_args()
    if args.target is None:
        answer = input("已解压的旧完整复现包根目录（包含 runtime 和 AnimalAct-SlowFast）: ").strip().strip('"')
        if not answer:
            parser.error("Target bundle directory is required")
        args.target = Path(answer)
    try:
        apply(args.target, dry_run=args.dry_run, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"UPDATE STOPPED: {error}", file=sys.stderr, flush=True)
        print("Existing backups are retained. Resolve the reported problem and rerun; do not delete your project.",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
