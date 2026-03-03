from setuptools import setup, find_packages

setup(
    name="pycmaetad",
    version="0.1.0",
    description="CMA-ES optimisation of metadynamics bias parameters",
    packages=find_packages(),
    python_requires=">=3.9",
    # Only pip-installable dependencies go here.
    # OpenMM, PLUMED, and the OpenMM-PLUMED plugin must be installed via conda
    # (see environment.yml) and are intentionally omitted.
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "cmaes",
        "pyyaml",
        "joblib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)