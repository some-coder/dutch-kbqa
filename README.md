# Dutch KBQA

This project stores Niels' master work project. It's about knowledge
base question answering (or KBQA for short).

## Run this project

Ensure you have the following:

- A computer with a UNIX terminal.
- A GitHub account with SSH access to it.
- Python 3 with the `venv` package installed.

First, download the repository and navigate to `development`.

```shell
# navigate to a directory in which to place this project
git clone git@github.com:some-coder/dutch-kbqa.git project
cd project
git checkout -b development
git pull origin development
```

Then set up Python.

```shell
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

Then run the project.

```shell
python3 main.py
```

## View the paper on which this work is based

See [this paper](https://arxiv.org/abs/2108.03509).

