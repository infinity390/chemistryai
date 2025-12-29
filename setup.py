from setuptools import setup, find_packages

setup(
    name="chemistryai",
    version="0.2.4",
    packages=find_packages(),
    install_requires=[
        "rdkit==2025.9.3",
        "Pillow>=10.0.0",
    ],
    python_requires='>=3.9',
    author="education_is_self_done",
    author_email="swastikmozumder@gmail.com",
    description="chemistry library tailored for iit-jee",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/infinity390/chemistryai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
