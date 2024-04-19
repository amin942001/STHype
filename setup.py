from setuptools import find_packages, setup

# with open("STHype/README.md", "r") as f:
#     long_description = f.read()

setup(
    name="sthype",
    version="0.0.1",
    description="Generate hypergrapsh from time correlated spatial graphs",
    # package_dir={"": "STHype"},
    packages=find_packages(where="."),
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/amin942001/STHype",
    license="MIT",
    install_requires=[""],
    extras_require={
        "dev": [],
    },
    python_requires=">3.7",
)
