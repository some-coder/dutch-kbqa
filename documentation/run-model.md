# Train, Validate, and Test the Transformer Model

Given a prepared dataset (see `documentation/create-dutch-dataset.md`), we are in the position to train, validate, and test a BERT-based transformer model. We will first discuss how to fine-tune a pre-trained transformer on our downstream Dutch-to-SPARQL task ('training' and 'validating'). Thereafter, we show how to test the trained-and-validated model.

## Train and validate the transformer

Before training and validating the transformer, we must, once again, prepare our Python environment. Do this by executing the following commands:

```sh
cd source/py-model/
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

Once done, go back to the project's root directory. Train and validate the model by calling

```sh
(set -a .env && source .env && ./shell-scripts/run-model/train.sh)
```

Once the process has finished, you can find your model under `${MODELS_DIR}/${MODEL_NAME}/`.

**Note.** Of course, you probably want to change the defaults inherited by copying from `.sample-env`. Edit values in the lower half of your `.env` file to manipulate training and validation (and also testing, discussed below).

## Test the trained-and-validated transformer

Simply call, within the project's root directory, the command

```sh
(set -a .env && source .env && ./shell-scripts/run-model/test.sh)
```

By default, testing results are written into `${MODELS_DIR}/${MODEL_NAME}-test/`.
