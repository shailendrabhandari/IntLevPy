from setuptools import setup, find_packages

# Combine README.md and AUTHORS.rst for the long description
with open("README.md", "r") as readme_file, open("AUTHORS.rst", "r") as authors_file:
    long_description = readme_file.read() + "\n\n" + authors_file.read()

setup(
    name="intermittent_levy",
    version="v0.2",
    author="Shailendra Bhandari",
    author_email="shailendra.bhandari@oslomet.no",
    description="A Python toolkit for simulating intermittent processes and Levy flights.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Keep as "text/markdown" for compatibility with PyPI
    url="https://github.com/shailendrabhandari/IntLevy-Processes",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
