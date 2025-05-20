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
        'pandas>=2.2.3',
        'PyYAML>=6.0.2',
        'numpy>=2.2.4',
        'matplotlib>=3.10.1',
        'seaborn>=0.13.2',
        'setuptools>=78.1.0',
        'plotly>=6.0.1',
        'bokeh>=3.7.3',
        'pyproj>=3.7.1',
        'requests>=2.32.3'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)