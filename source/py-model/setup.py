"""Sets up the `py-model` Python sub-project."""

from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(name='dutch-kbqa-py-model',
          version='0.1.0',
          author='Niels de Jong',
          author_email='n.a.de.jong@student.rug.nl',
          description='Symbols for fine-tuning a BERT-based transformer ' +
                      'for Dutch WikiData question-answering.',
          url='https://github.com/some-coder/dutch-kbqa',
          packages=find_packages())
	
