"""
setup.py for DLVT (Dynamic Leadership Vitality Theory) package.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dlvt',
    version='1.0.0',
    author='William Bendinelli',
    author_email='',
    description='Dynamic Leadership Vitality Theory: A dynamical systems model of executive sustainability',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/dlvt',
    license='MIT',
    packages=find_packages(exclude=['notebooks', 'scripts', 'tests']),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'matplotlib>=3.7',
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0',
            'ipykernel>=6.0',
            'pytest>=7.0',
            'black>=22.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords='dynamical-systems leadership executive burnout complexity',
    project_urls={
        'Source': 'https://github.com/yourusername/dlvt',
        'Paper': 'https://doi.org/your.paper.doi',
    },
)
