from setuptools import find_packages, setup


def parse_requirements(path):
    def _parse_requirements(path):
        with open(path) as fin:
            return [line.strip() for line in fin]

    with open(path) as fin:
        reqs = []
        for line in fin:
            if line.startswith("-r "):
                _, reqs_path = line.split(" ")
                reqs.extend(_parse_requirements(reqs_path.strip()))
            else:
                reqs.append(line.strip())

    return reqs


setup(
    name="clfit",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Kirill Brodt",
    install_requires=parse_requirements("./requirements/runtime.txt"),
    extras_require={
        "all": parse_requirements("./requirements.txt"),
        "optional": parse_requirements("./requirements/optional.txt"),
        "interactive": parse_requirements("./requirements/interactive.txt"),
        "tests": parse_requirements("./requirements/tests.txt"),
    },
    license="GPL",
)
