from setuptools import setup, find_packages

setup(
    name="intermittent_levy",
    version="0.1",
    author="Shailendra Bhandari",
    author_email="shailendra.bhandari@oslomet.no",
    description="A Python toolkit for simulating intermittent processes and Levy flights.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shailendrabhandari/IntLevy-Processes",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
