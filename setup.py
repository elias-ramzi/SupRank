import setuptools

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name='suprank',
    version='1.0.0',
    packages=[''],
    url='https://github.com/elias-ramzi/SupRank',
    license='MIT',
    author='Elias Ramzi',
    author_email='elias.ramzi@valeo.com',
    description='This repo contains the official PyTorch implementation of our paper: Optimization of Rank Losses for Image Retrieval (https://arxiv.org/abs/2207.04873).',
    python_requires='>=3.6',
    install_requires=install_requires
)
