from setuptools import setup, find_packages

with open("README.md", "r") as f:
  long_description = f.read()

setup(
    name="curator-torch",
    version="1.1.1",
    short_description="Library for implementation of message passing neural networks in Pytorch",
    long_description=long_description,
    author="xinyang",
    author_email="xinyang@dtu.dk",
    license='MIT License',
    url = "https://github.com/dtu-energy/curator.git",
    packages=find_packages(include=["curator", "curator.*"]),
    scripts=[
        "scripts/curator-train",
        "scripts/curator-tmptrain",
        "scripts/curator-simulate",
        "scripts/curator-select",
        "scripts/curator-label",
        "scripts/curator-deploy",
        "scripts/curator-workflow",
    ],
    install_requires=[
        'myqueue',
        'torch>=1.10',
        'ase',
        'asap3',
        'e3nn',
        'torch-ema>=0.3.0',
        'toml',
        'numpy',
        'hydra-core>=1.2',
        'wandb',
        'lightning',
    ],
    include_package_data=True,
)
