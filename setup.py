from setuptools import setup, find_packages

setup(
    name="chemistryai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rdkit-pypi",
    ],
    python_requires='>=3.9',
    author="education_is_self_done",
    author_email="swastikmozumder@gmail.com",
    description="Simple chemical structure parser, IUPAC namer, and visualizer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/infinity390/chemistryai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
