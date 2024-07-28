from setuptools import setup, find_packages

setup(
    name="mindex",
    version="0.1.0",
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=[
        'numpy==1.26.4',
        'torch==2.3.0',
        'sentence-transformers==3.0.0'
    ],
    author="Leon",
    author_email="leon.ericsson@icloud.com",
    description="A local semantic search engine over your mind index.",
    url="https://github.com/LeonEricsson/mindex",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)