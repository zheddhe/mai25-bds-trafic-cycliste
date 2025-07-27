# -*- coding: utf-8 -*-
import nox  # type: ignore
import shutil
import re
import subprocess
from pathlib import Path

PYTHON_VERSION = "3.12"
PYTHON_VERSION_DL_TF = "3.9"


def remove_paths(session, paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            session.log(f"Removed {path}")
    for pyc in Path(".").rglob("*.pyc"):
        pyc.unlink()
    for cover in Path(".").rglob("*,cover"):
        cover.unlink()
    for cache in Path(".").rglob("__pycache__"):
        shutil.rmtree(cache)


def get_cuda_toolkit_version():
    """Detect CUDA runtime version via nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
        match = re.search(r"CUDA Version\s*:\s*([\d.]+)", output)
        if match:
            version_str = match.group(1)
            major_minor = version_str.split(".")[:2]
            return float(".".join(major_minor))
    except Exception:
        return None


@nox.session(python=PYTHON_VERSION)
def clean_project(session):
    """Remove temporary files and build artifacts (cross-platform, without .nox)."""
    paths = [
        ".pytest_cache", ".coverage", "htmlcov",
        "build", "dist", "*.egg-info"
    ]
    remove_paths(session, paths)


@nox.session(python=PYTHON_VERSION)
def clean_all(session):
    """Remove all temporary files, including .nox environments."""
    paths = [
        ".pytest_cache", ".coverage", "htmlcov",
        "build", "dist", ".nox", "*.egg-info"
    ]
    remove_paths(session, paths)


@nox.session(python=PYTHON_VERSION, venv_backend="conda",
             name=f"build-{PYTHON_VERSION}")
def build(session):
    """Run code linting and full test suite with coverage and HTML report."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)
    session.install("-e", ".[py312, test, dev, dl]", silent=False)
    cuda_version = get_cuda_toolkit_version()
    session.log(f"Detected CUDA runtime version: {cuda_version}")
    if cuda_version is None:
        session.info("Unable to detect CUDA version (nvidia-smi missing?)")

    if cuda_version and cuda_version >= 12.8:
        cu_tag = "+cu128"
    elif cuda_version and (11.8 <= cuda_version < 12.0):
        cu_tag = "+cu118"
    else:
        cu_tag = ""
        session.info(f"Unsupported CUDA version: {cuda_version},"
                     " falling back to CPU builds for pytorch packages")
    # Torch GPU via wheels cu118
    session.install(
        f"torch==2.7.1{cu_tag}",
        "-f", "https://download.pytorch.org/whl/torch/"
    )
    # TorchVision GPU via wheels cu118
    session.install(
        f"torchvision==0.22.1{cu_tag}",
        "-f", "https://download.pytorch.org/whl/torchvision/"
    )
    # TorchAudio GPU via wheels cu118
    session.install(
        f"torchaudio==2.7.1{cu_tag}",
        "-f", "https://download.pytorch.org/whl/torchaudio/"
    )
    session.run("flake8")
    session.run("pytest")
    session.log("Build session complete. Coverage report in htmlcov/index.html")


@nox.session(python=PYTHON_VERSION, venv_backend="conda",
             name=f"package-{PYTHON_VERSION}")
def package(session):
    """Package the project (sdist + wheel)."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)
    session.install("build")
    session.run("python", "-m", "build")
    session.log("Package session complete.")


@nox.session(python=PYTHON_VERSION_DL_TF, venv_backend="conda",
             name=f"dl-tensorflow-{PYTHON_VERSION_DL_TF}")
def deep_learning_tf(session):
    """DL session for TensorFlow GPU with compatible Python."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)
    # Installation depuis conda, build GPU officielle
    session.conda_install(
        "-c", "pytorch",
        "-c", "defaults",
        "tensorflow=2.10.0=gpu_py39h9bca9fa_0",
    )
    session.install("-e", ".[py39, test, dev]", silent=False)
    session.run(
        "python", "-c", "import tensorflow as tf; "
        "print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
    )
    session.log("TensorFlow GPU environment ready.")
