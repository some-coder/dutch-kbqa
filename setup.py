"""
Contains a method for creating a local Python package of this project.
"""


from setuptools import setup


if __name__ == '__main__':
	setup(
		name='dutch_kbqa',
		version='1.0',
		author='Niels de Jong',
		author_email='n.a.de.jong@student.rug.nl',
		description='A Python package for creating a Dutch KBQA system.',
		license='MIT',
		url='https://github.com/some-coder/dutch-kbqa',
		packages=['pre_processing', 'utility']
	)
