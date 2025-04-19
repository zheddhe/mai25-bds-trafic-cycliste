# optional for installation as a package
from setuptools import setup, find_packages

setup(
    name='avr25-mle-velib',
    version='0.1.0',
    description='velib data science project for april 2025 promotion',
    author='Rémy CANAL',
    author_email='remy.canal@live.fr',
    packages=find_packages(where='.'),  # find all packages in the current directory
    install_requires=[
        'pandas>=2.0',
        'PyYAML>=6.0',
        'numpy>=1.24',
        'matplotlib>=3.5'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)