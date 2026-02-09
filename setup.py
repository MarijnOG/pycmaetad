from setuptools import setup, find_packages

setup(
    name="pycmaetad",
    version="0.1.0",
    description="CMA-ES optimization of metadynamics bias parameters",
    author="Human Person",
    author_email="Human.Person@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[ 
        #TODO: specify dependencies
    ],
    # TODO: add optional dependencies
    # extras_require={
    #     "dev": [],
    #     "viz": [],
    # },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)