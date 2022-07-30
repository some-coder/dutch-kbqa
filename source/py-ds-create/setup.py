"""Sets up the `py_ds_create` Python sub-project."""

from setuptools import setup, find_packages


if __name__ == '__main__':
	setup(name='dutch-kbqa-py-ds-create',
	      version='0.1.0',
	      author='Niels de Jong',
				author_email='n.a.de.jong@student.rug.nl',
				description='Symbols for creating a Dutch derivation of the ' +
				            'LC-QuAD 2.0 dataset.',
				url='https://github.com/some-coder/dutch-kbqa',
				packages=find_packages())
	
