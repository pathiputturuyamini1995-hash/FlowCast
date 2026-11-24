from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_project_folders_exist():
    assert (PROJECT_DIR / "data").exists()
    assert (PROJECT_DIR / "models").exists()
    assert (PROJECT_DIR / "dashboard").exists()


def test_requirements_file_exists():
    assert (PROJECT_DIR / "requirements.txt").exists()