from pathlib import Path

from setuptools import find_packages, setup

long_description = Path("README.md").read_text().strip()

setup(
    name="H2iPort_public",
    version="0.9.0",
    author="",
    description="Hydrogen infrastructure model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hka-esa/H2iPort_public",
    license="LGPLv3",
    packages=find_packages(exclude=["doc"]),
    python_requires=">=3.7",
    install_requires=["pandas", "ruamel.yaml"],
    extras_require={"dev": ["black", "isort", "pytest"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["hydrogen", "energy systems"],
)
