"""
This setup.py allows your python package to be installed. 
Please completely update the parameters in the opts dictionary. 
For more information, see https://stackoverflow.com/questions/1471994/what-is-setup-py
"""

from setuptools import setup, find_packages
PACKAGES = find_packages()

"""Read the contents of your README file"""
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

opts = dict(name='betabinom_test',
            maintainer='Andrew Fiore-Gartland',
            maintainer_email='agartlan@fredhutch.org',
            description='Bootstrap testing for a beta-binomial regression model.',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/agartland/betabinom_test',
            license='MIT',
            author='Andrew Fiore-Gartland',
            author_email='agartlan@fredhutch.org',
            version='0.1',
            packages=PACKAGES,
            include_package_data=True
           )

install_reqs = [  'numpy',
                  'pandas',
                  'scipy',
                  'patsy',
                  'parmap']

if __name__ == "__main__":
    setup(**opts, install_requires=install_reqs)

