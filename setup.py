from setuptools import find_packages, setup

with open("requirements.txt") as fin:
    install_req = [line.strip() for line in fin]

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Kirill Brodt",
    install_requires=install_req,
    license="GPL",
)
