from setuptools import setup, find_packages

setup(
        name="FINER", 
        version='1.0.0',
        author="Ke Jin and Liwei Wang",
        author_email="kej13@mails.ccnu.edu.cn, wangliwei@mails.ccnu.edu.cn",
        description='FINER: Framework for Identity- and Niche-informed Expression Reconstruction',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            ],
        classifiers= [
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
        ],
        python_requires='>=3.8',
)



