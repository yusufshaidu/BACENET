from setuptools import setup, find_packages

setup(
    name="BACENET",                        # your project name
    version="0.1.0",                        # initial version
    description="BACE-Net: Behler-Parrinello Atomic Cluster Expansion neural Networks",
    author="Yusuf Shaidu",
    author_email="yusufshaidu@gmail.com",
    url="https://github.com/yusufshaidu/silBP.git",  # your repo URL
    packages=find_packages(exclude=["tools*", "examples*"]),
    python_requires=">=3.7, <4",
    install_requires=[
        #GPU enabled tensorflow==14.0 requires cuda 11.7 and cuDNN 8.9 
        "numpy>=1.19",
        "tensorflow>2.14, <=2.15.0",
        "keras<=2.15.0",
        "keras-swa>=0.0.8",
        "ase>=3.20",
        "pyyaml>=5.1",
    ],
    entry_points={
        "console_scripts": [
            # replace `train.py`'s `if __name__=='__main__': ...` 
            # with a `main()` function or point it here
            "bacenet-train = bacenet.train:main",
            "bacenet-evaluate = bacenet.evaluate:main",
            "bacenet-ase = bacenet.ase_interface:wBP_Calculator",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    zip_safe=False,
)

