from pathlib import Path


def test_no_forbidden_application_dependencies_or_imports():
    root = Path(__file__).resolve().parents[1]
    forbidden = ["net" + "-hdbscan", "net" + "-center", "net" + "-voronoi"]
    targets = [root / "pyproject.toml", root / "README.md"]
    targets += sorted((root / "src").rglob("*.py"))
    targets += sorted((root / "tests").rglob("*.py"))
    for path in targets:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"forbidden dependency boundary leaked into {path}: {token}"
