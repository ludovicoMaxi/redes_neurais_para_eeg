from setuptools import setup, find_packages

setup(
    name="eeg_ia",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "tensorflow"
    ],
)