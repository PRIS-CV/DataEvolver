"""Regression cases for source checks; no application or native worker imports."""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / '.agents/skills/verify-dataevolver/scripts/check_repository.py'
SPEC = importlib.util.spec_from_file_location('repository_contracts', SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_broken_imports_and_retired_entrypoints_are_reported(tmp_path):
    package = tmp_path / 'src/dataevolver/harness'
    package.mkdir(parents=True)
    cli = package / 'cli.py'
    cli.write_text('from dataevolver.removed import worker\ndef main(): pass\n')
    (tmp_path / 'pyproject.toml').write_text('''[project.scripts]
dataevolver = "dataevolver.harness.cli:main"
dataevolver-harness = "dataevolver.harness.cli:main"
dataevolver-production = "dataevolver.cli.production:main"
''')
    errors = CHECKER.check(tmp_path)
    assert any('missing internal module dataevolver.removed' in error for error in errors)
    assert any('has no source function' in error for error in errors)
    assert any('must not bypass' in error for error in errors)
    cli.write_text('def main(): pass\n')
    config = tmp_path / 'pyproject.toml'
    config.write_text('\n'.join(line for line in config.read_text().splitlines() if not line.startswith('dataevolver-production')))
    assert CHECKER.check(tmp_path) == []


def test_private_machine_paths_are_reported(tmp_path):
    (tmp_path / 'pyproject.toml').write_text('''[project.scripts]
dataevolver = "dataevolver.harness.cli:main"
dataevolver-harness = "dataevolver.harness.cli:main"
''')
    package = tmp_path / 'src/dataevolver/harness'
    package.mkdir(parents=True)
    (package / 'cli.py').write_text('def main(): pass\n')
    path = tmp_path / 'config.txt'
    path.write_text('/Users/' + 'fixture/private-model')
    assert any('private path or credential' in error for error in CHECKER.check(tmp_path))
    path.write_text('model_path_env=HOST_MODEL_ROOT')
    assert CHECKER.check(tmp_path) == []
