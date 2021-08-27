#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

from setuptools import Extension, find_packages
from skbuild import setup
from setuptools.command.build_ext import build_ext


setup(
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(where = 'src'),
    package_dir={"": "src"},
    cmake_install_dir=os.path.join("src", "simsoptpp")
)
