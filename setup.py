from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name='seqrep',
    url='https://github.com/MIR-MU/seqrep',
    author='Jakub Rysavy',
    author_email='jakubrysavy00@gmail.com',
    packages=['seqrep'],
    install_requires=['pandas'],
    version='0.0.0',
    license='MIT',
    description='Scientific framework for representation in sequential data',
    long_description=readme,
)