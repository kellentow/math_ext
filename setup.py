# filepath: /home/kellen/Coding/Linux/python/math_ext/setup.py
from setuptools import setup, find_packages

setup(
    name='math_ext',
    version='0.2',
    packages=find_packages(),
    install_requires=[],
    author='Kellen Townsend',
    author_email='kellencoding@icloud.com',
    description='Adds more math functions to python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kellentow/math_ext',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Mozilla License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3',
)