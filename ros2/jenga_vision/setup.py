"""ROS 2 (ament_python) build of the jenga_vision library in ../../src.

jenga_vision/, config/ and test/ here are symlinks into the uv project at the
repo root, so both builds use the same code. The package sits in this
subdirectory because setuptools >= 61 (Jazzy) would take the name, version
and scripts from the root pyproject.toml instead of this file.

Executables are the [project.scripts] of pyproject.toml, with ROS 2 names:
jenga-autoscan -> ros2 run jenga_vision jenga_autoscan.
"""

import re
from glob import glob
from pathlib import Path

from setuptools import setup

package_name = "jenga_vision"
pyproject = (Path(__file__).resolve().parent / "../../pyproject.toml").read_text()
scripts = re.search(r"^\[project\.scripts\]\n(.*?)(?:^\[|\Z)", pyproject, re.M | re.S).group(1)
console_scripts = [f"{name.replace('-', '_')} = {target}"
                   for name, target in re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', scripts, re.M)]

setup(
    name=package_name,
    version=re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1),
    packages=[package_name],
    package_data={package_name: ["dashboard.html", "py.typed"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.json") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="wonik",
    maintainer_email="car991231@gmail.com",
    description="Dino-Lite Jenga scan: end-face gaps, pull order and per-block heights for jenga_game",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": console_scripts},
)
