from pathlib import Path

import tomllib


def test_official_sam_compatibility_dependencies_are_pinned() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    build_dependencies = project["build-system"]["requires"]
    runtime_dependencies = project["project"]["dependencies"]
    all_dependencies = project["project"]["optional-dependencies"]["all"]

    assert "setuptools>=75,<82" in build_dependencies
    assert "setuptools>=75,<82" in runtime_dependencies
    assert "einops>=0.8,<1" in runtime_dependencies
    assert "pycocotools>=2.0.7,<3" in runtime_dependencies
    assert "psutil>=5.9,<8" in runtime_dependencies
    assert "numpy>=1.26,<2" in runtime_dependencies
    assert "scipy<1.18" in runtime_dependencies
    assert "opencv-python-headless==4.11.0.86" in all_dependencies
