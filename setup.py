import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="MiTgaze",
    version="0.0.3",
    url="https://gitlab.tuebingen.mpg.de/vbokharaie/mitgaze/",
    license="GNU",
    author="Vahid Samadi Bokharaie",
    author_email="vahid.bokharaie@protonmail.com",
    description="A python-based tool to study gaze behaviour in Eye-Tracking Experiments. ",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests", "venv")),
    test_suite="nose.collector",
    tests_require=["nose"],
    package_data={"mitgaze": ["datasets/*.*"]},
    include_package_data=True,
    install_requires=[
        "seaborn",
        "pathlib",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "openpyxl",
        "scikit-image",
        "joblib",
        "opencv-python",
        "pillow",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GPLv3",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
