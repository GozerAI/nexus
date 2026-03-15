"""Setup configuration for Nexus package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nexus-ai",
    version="1.0.0",
    author="Christopher R. Arsenault",
    author_email="chris@gozerai.com",
    description="Advanced AI Ensemble, Orchestrator & Consciousness Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chrisarseno/Nexus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nexus-cli=nexus.cli.main:cli",
            "nexus-chat=nexus.cli.chat:main",
            "nexus-api=nexus.api.api:main",
            "nexus-gui=nexus.gui:run",
        ],
    },
)
