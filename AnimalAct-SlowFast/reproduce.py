"""Print or execute a portable training recipe, always into a new output directory."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    command = [sys.executable, str(project / recipe['entrypoint'])]
    for key, value in recipe['arguments'].items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            command.append(flag)
            if isinstance(value, list):
                command.extend(str(item) for item in value)
            else:
                command.append(str(value))
    command.extend(['--out', str(args.out.resolve())])
    print(subprocess.list2cmdline(command), flush=True)
    if args.execute:
        if args.out.exists():
            parser.error('Output exists; use a new path to preserve experiments.')
        subprocess.run(command, cwd=project, check=True)


if __name__ == '__main__':
    main()
