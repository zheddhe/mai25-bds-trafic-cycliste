import nox # type: ignore
import shutil
from pathlib import Path

# Python version(s) to use for test sessions
PYTHON_VERSIONS = ["3.13"]

def remove_paths(session, paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            session.log(f"🧹 Removed {path}")

    for pyc in Path(".").rglob("*.pyc"):
        pyc.unlink()
    for cover in Path(".").rglob("*,cover"):
        cover.unlink()
    for cache in Path(".").rglob("__pycache__"):
        shutil.rmtree(cache)

@nox.session(python=PYTHON_VERSIONS)
def clean_project(session):
    """Remove temporary files and build artifacts (cross-platform, without .nox)."""
    paths = [
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
        "avr25_mle_velib.egg-info",
    ]
    remove_paths(session, paths)

@nox.session(python=PYTHON_VERSIONS)
def clean_all(session):
    """Remove all temporary files, including .nox environments."""
    paths = [
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
        ".nox",
        "avr25_mle_velib.egg-info",
    ]
    remove_paths(session, paths)

@nox.session(python=PYTHON_VERSIONS)
def build(session):
    """Build the project (sdist + wheel)."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)
    session.install("build")
    session.run("python", "-m", "build")
    session.log("✅ Build session complete.")

@nox.session(python=PYTHON_VERSIONS)
def full(session):
    """Run full test suite with coverage and HTML report in a single session."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)
    session.install(".", ".[dev]")
    session.run("pytest")
    session.log("✅ Full session complete. Coverage report in htmlcov/index.html")
