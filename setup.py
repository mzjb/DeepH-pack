import os.path
import codecs
from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="deeph",
    version=get_version("deeph/__init__.py"),
    description="DeepH-pack is the official implementation of the Deep Hamiltonian (DeepH) method.",
    download_url="https://github.com/mzjb/DeepH-pack",
    author="He Li",
    python_requires=">=3.9",
    packages=find_packages(),
    package_dir={'deeph': 'deeph'},
    package_data={'': ['*.jl', '*.ini', 'periodic_table.json']},
    entry_points={
        "console_scripts": [
            "deeph-preprocess = deeph.scripts.preprocess:main",
            "deeph-train = deeph.scripts.train:main",
            "deeph-evaluate = deeph.scripts.evaluate:main",
            "deeph-inference = deeph.scripts.inference:main",
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "torch>=1.9",
        "torch_geometric>=1.7.2",
        "e3nn>=0.3.5, <=0.4.4",
        "h5py",
        "pymatgen",
        "pathos",
        "psutil",
        "tqdm",
        "tensorboard",
    ],
    license="MIT",
    license_files="LICENSE",
    zip_safe=False,
)
