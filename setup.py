from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

import versioneer

NAME = "silicone"
SHORT_DESCRIPTION = "Automated filling of detail in reported emission scenarios"
KEYWORDS = ["emissions", "automation", "filling", "detail", "climate"]
AUTHORS = [
    ("Robin Lamboll", "r.lamboll@imperial.ac.uk"),
    ("Zebedee Nicholls", "zebedee.nicholls@climate-energy-college.org"),
    ("Jarmo Kikstra", "kikstra@iiasa.ac.at"),
    ("Gaurav Ganti", "gaurav.ganti@climateanalytics.org"),
]
URL = "https://github.com/GranthamImperial/silicone"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/GranthamImperial/silicone/issues",
    "Documentation": "https://silicone.readthedocs.io/en/latest",
    "Source": "https://github.com/GranthamImperial/silicone",
}
README = "README.rst"

SOURCE_DIR = "src"

LICENSE = "3-Clause BSD License"
CLASSIFIERS = [
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

REQUIREMENTS_INSTALL = [
    "openscm-units>=0.2",
    "pint",
    "pyam-iamc>=1.0.0",
    "python-dateutil",
    "numpy",
    "scipy",
    "tqdm",
]
REQUIREMENTS_NOTEBOOKS = ["notebook", "seaborn", "statsmodels"]
REQUIREMENTS_TESTS = [
    "codecov",
    "nbval",
    "coverage",
    "pytest",
    "pytest-console-scripts",
    "pytest-cov",
]
REQUIREMENTS_DOCS = [
    "sphinx>2.1",
    "sphinx_rtd_theme",
    "sphinx-autodoc-typehints",
]
REQUIREMENTS_DEPLOY = ["setuptools>=38.6.0", "twine>=1.11.0", "wheel>=0.31.0"]
REQUIREMENTS_DEV = (
    [
        "black",
        "black-nb",
        "bandit",
        "coverage",
        "flake8",
        "isort",
        "mypy",
        "nbdime",
        "pydocstyle",
        "pylint",
    ]
    + REQUIREMENTS_NOTEBOOKS
    + REQUIREMENTS_TESTS
    + REQUIREMENTS_DOCS
    + REQUIREMENTS_DEPLOY
)


REQUIREMENTS_EXTRAS = {
    "notebooks": REQUIREMENTS_NOTEBOOKS,
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": REQUIREMENTS_DEV,
}

# Get the long description from the README file
with open(README, "r") as f:
    README_LINES = ["Silicone", "========", ""]
    add_line = False
    for line in f:
        if line.strip() == ".. sec-begin-long-description":
            add_line = True
        elif line.strip() == ".. sec-end-long-description":
            break
        elif add_line:
            README_LINES.append(line.strip())

if len(README_LINES) < 3:
    raise RuntimeError("Insufficient description given")


class SiliconeTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


CMDCLASS = versioneer.get_cmdclass()
CMDCLASS.update({"test": SiliconeTest})

setup(
    name=NAME,
    version=versioneer.get_version(),
    description=SHORT_DESCRIPTION,
    long_description="\n".join(README_LINES),
    long_description_content_type="text/x-rst",
    keywords=KEYWORDS,
    author=", ".join([author[0] for author in AUTHORS]),
    author_email=", ".join([author[1] for author in AUTHORS]),
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(SOURCE_DIR),
    package_dir={"": SOURCE_DIR},
    install_requires=REQUIREMENTS_INSTALL,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=CMDCLASS,
)
