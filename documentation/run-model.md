# Train, Validate, and Test the Transformer Model

Given (1) a prepared dataset (see `documentation/create-dutch-dataset.md`) and (2) a completely initialised Conda environment (see `documentation/install-model-dependencies.md`), we are in the position to train, validate, and test a BERT-based transformer model. We will first discuss how to fine-tune a pre-trained transformer on our downstream Dutch-to-SPARQL task ('training' and 'validating'). Thereafter, we show how to test the trained-and-validated model.

## Train and validate the transformer

In the project's root directory, train and validate the model by calling

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
