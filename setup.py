from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="seqrep",
    url="https://github.com/MIR-MU/seqrep",
    author="Jakub Rysavy",
    author_email="jakubrysavy00@gmail.com",
    packages=["seqrep"],
    install_requires=[
        "hrv_analysis>=1.0.4",
        "numpy_ext>=0.9.6",
        "pandas>=1.1.5",
        "pandas_ta>=0.3.14b0",
        "plotly>=4.4.1",
        "scikit_learn>=1.0.1",
        "ta>=0.8.0",
        "tqdm>=4.62.3",
    ],
    tests_require=["pytest"],
    version="0.0.2",
    license="MIT",
    description="Scientific framework for representation in sequential data",
    long_description=readme,
    long_description_content_type="text/markdown",
)