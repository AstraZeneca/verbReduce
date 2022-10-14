import nltk
import spacy
from setuptools import find_packages, setup


def read_requirements(filename: str):
    # Install NLTK dependecies
    nltk.download("wordnet")
    nltk.download("stopwords")

    # Install Spacy dependecies
    spacy.download("en_core_web_sm")
    with open(filename) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git",
                req,
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))
    return requirements


VERSION = 0.01

setup(
    name="verb_cluster",
    version=VERSION,
    description="Reducing the verb cardinality using a Self-Supervised Transformer model with SAT reduction",
    keywords="self-supervised,SAT",
    license="Apache",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.9",
)
