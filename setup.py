from setuptools import setup
import setuptools

with open("README.md", 'r') as file:
    long_description = file.read()
setup(
    name='recommender-blackboxes',
    version='0.0.01',
    author_email=' rp813149@gmail.com',
    description='Easy to use Recommender system package',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        'numpy>=1.18.5',
        'pandas>=1.0.5',
        'scikit_learn>=0.23.2',
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
)
