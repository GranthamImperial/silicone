import versioneer
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

NAME = "silicone"
SHORT_DESCRIPTION = "Automated filling of detail in reported emission scenarios"
KEYWORDS = ["emissions", "automation", "filling", "detail", "climate"]
AUTHORS = [
    ("Zeb Nicholls", "zebedee.nicholls@climate-energy-college.org"),
]

README = "README.rst"
URL = "https://github.com/znicholls/silicone"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/znicholls/silicone/issues",
    "Documentation": "https://silicone.readthedocs.io/en/latest",
    "Source": "https://github.com/znicholls/silicone",
}

SOURCE_DIR = "src"

# LICENSE = "GNU Affero General Public License v3.0 or later"
CLASSIFIERS = [
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    # "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
]
REQUIREMENTS_INSTALL = ["numpy", "scipy", "pint", "python-dateutil", "pyam-iamc>=0.2.0"]
REQUIREMENTS_NOTEBOOKS = [
    "matplotlib",
    "notebook",
    "seaborn",
]
REQUIREMENTS_TESTS = [
    "codecov",
    "nbval",
    "pytest",
    "pytest-cov",
]
REQUIREMENTS_DOCS = ["sphinx>=1.8", "sphinx_rtd_theme", "sphinx-autodoc-typehints"]
REQUIREMENTS_DEPLOY = ["setuptools>=38.6.0", "twine>=1.11.0", "wheel>=0.31.0"]
REQUIREMENTS_DEV = (
    [
        "black",
        "bandit",
        "coverage",
        "flake8",
        "isort",
        "mypy",
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
with open(README, "r", encoding="utf-8") as f:
    README_LINES = ["Silicone", "========", ""]
    for line in f:
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
    # license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(SOURCE_DIR),
    package_dir={"": SOURCE_DIR},
    install_requires=REQUIREMENTS_INSTALL,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=CMDCLASS,
)
