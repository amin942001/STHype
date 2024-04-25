"""Setup for the sthype package"""

from setuptools import find_packages, setup

with open("sthype/README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sthype",
    version="0.0.1",
    description="Generate hypergraph from time correlated spatial graphs",
    packages=find_packages(where="."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amin942001/STHype",
    license="MIT",
    install_requires=["matplotlib", "networkx", "numpy", "shapely"],
    extras_require={
        "dev": [],
    },
    python_requires=">3.7",
)
