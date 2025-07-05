# -*- coding: utf-8 -*-
import nox  # type: ignore
import shutil
from pathlib import Path

PYTHON_VERSION = "3.12"
PYTHON_VERSION_DL = "3.9"


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
    session.install("-e", ".[py312, dev]", silent=False)
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


@nox.session(python=PYTHON_VERSION_DL, venv_backend="conda",
             name=f"dl-tensorflow-{PYTHON_VERSION_DL}")
def deep_learning_tf(session):
    """DL session for TensorFlow GPU with Python 3.9."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)

    # Installation depuis conda, build GPU officielle
    session.conda_install(
        "-c", "pytorch",
        "-c", "defaults",
        "tensorflow=2.10.0=gpu_py39h9bca9fa_0",
    )

    session.install("-e", ".[py39, dev]")

    session.run(
        "python", "-c", "import tensorflow as tf; "
        "print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
    )
    session.log("TensorFlow GPU environment ready.")


@nox.session(python=PYTHON_VERSION_DL, venv_backend="conda",
             name=f"dl-torch-{PYTHON_VERSION_DL}")
def deep_learning_torch(session):
    """DL session for PyTorch GPU with Python 3.9."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip", silent=True)

    # PyTorch GPU via wheels cu118
    session.install(
        "torch==2.3.1+cu118",
        "torchvision==0.18.1+cu118",
        "torchaudio==2.3.1+cu118",
        "-f", "https://download.pytorch.org/whl/torch_stable.html"
    )

    # Additional torch specifics that needs to be installed knowing where to find PyTorch
    session.install(
        "tensorboard",
        "torchsummary",
        "captum",
    )

    session.install("-e", ".[py39, dev]")

    session.run("python", "-c", "import torch; "
                "print('Torch CUDA:', torch.cuda.is_available())")
    session.log("PyTorch GPU environment ready.")
