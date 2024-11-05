# setup.py

from setuptools import setup, find_packages

setup(
    name='intermittent_levy',
    version='0.1',
    description='Package for classifying intermittent and Lévy processes',
    author='Shailendra Bhandari & Pedro Lencastre',
    author_email='shailendra.bhandari@oslomet.no',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'pomegranate',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)

