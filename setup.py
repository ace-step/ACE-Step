from setuptools import setup, find_namespace_packages

setup(
    name="ace_step",
    description="ACE Step: A Step Towards Music Generation Foundation Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version="0.1.2",
    packages=find_namespace_packages(),
    install_requires=open("requirements.txt", encoding="utf-8").read().splitlines(),
    author="ACE Studio, StepFun AI",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
    ],
    entry_points={
        "console_scripts": [
            "acestep=acestep.gui:main",
        ],
    },
    include_package_data=True,  # Ensure this is set to True
    package_data={
        "acestep.models.lyrics_utils": ["vocab.json"],  # Specify the relative path to vocab.json
    },
    extras_require={
        "train": [
            "peft",
            "tensorboard",
            "tensorboardX"
        ]
    },
)
