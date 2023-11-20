from setuptools import setup, find_packages

setup(
        name='offlinerlkit',
        version="0.0.1",
        description=(
            'OfflineRL-kit'
        ),
        author='Yihao Sun',
        author_email='sunyh@lamda.nju.edu.cn',
        maintainer='yihaosun1124',
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            "gym>=0.23.1,<=0.24.1",
            "matplotlib",
            "numpy",
            "pandas",
            "ray==1.13.0",
            "torch",
            "tensorboard",
            "tqdm",
        ]
    )
