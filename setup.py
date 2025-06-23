from setuptools import setup, find_packages

setup(
    name="tidymut",
    version="0.1.0",
    description="A biological sequence manipulation library",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=["pandas"],
    extras_require={
        "test": [
            "pytest>=6.0",
            "pytest-cov",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
)
