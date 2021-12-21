# Dutch KBQA

This project stores Niels' master work project. It's about knowledge
base question answering (or KBQA for short).

## Run this project

Ensure you have the following:

1. A computer with:
	1. a UNIX terminal, and
	2. (recommended) a GPU that can be found by TensorFlow and PyTorch.
2. A GitHub account with SSH access to it.
3. Python 3 with the `venv` package installed.

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
cd model
python3 -m venv .venv
source .venv/bin/activate
pip3 install wheel  # to avoid some problems
pip3 install -r requirements.txt
chmod +x fine-tune.sh  # you might need sudo capabilities for this
```

Then run the project.

```shell
./fine-tune.sh
```

This will train the BERT encoder-decoder model on the Dutch KBQA dataset. On Peregrine, this will take around
1.5 days, so be sure you want to wait this long if you run this on your local PC.

## Credits

See [this paper (Cui et al. 2021)](https://arxiv.org/abs/2108.03509).

For the code, we rely on [(multilingual) BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805) and
the code for the [SPBERT model (Tran et al. 2021)](https://arxiv.org/abs/2106.09997) (which we adapt for the Dutch
KBQA task).

Lastly, we thank the Center for Information Technology (CIT) of the University of Groningen for their support and for
providing access to the _Peregrine_ high performance computing cluster.
