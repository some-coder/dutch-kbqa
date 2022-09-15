# Train, Validate, and Test the Transformer Model

Given (1) a prepared dataset (see `documentation/create-dutch-dataset.md`) and (2) a completely initialised `pip3` or `conda` environment (see `documentation/install-model-dependencies.md`), we are in the position to train, validate, and test a BERT-based transformer model. We will first discuss how to fine-tune a pre-trained transformer on our downstream Dutch-to-SPARQL task ('training' and 'validating'). Thereafter, we show how to test the trained-and-validated model.

## Train and validate the transformer

Choose one of the preset environments from `environments/original` or `environments/appendix`, or define your own experiment by copying `.sample-env` and naming it something else, such as `.env`. Let your targeted environment be given the shorthand `$MY_ENV`.

First, ensure that two values in `$MY_ENV` have been set correctly:

1. `$PROJECT_ROOT`. It must be set to the _absolute_ (not relative) path of this repository's root.
2. `$PY_MODEL_PACKAGE_MANAGER`. This environment variable must be set to `"pip"` (no `"3"` at the end!) if you used `pip3` as your dependency manager during the steps described in `documentation/install-model-dependencies.md`; if you used `conda` instead, set the variable to `"conda"`.

In the project's root directory, train and validate the model by calling

```sh
(set -a $MY_ENV && source $MY_ENV && ./shell-scripts/run-model/train.sh)
```

Once the process has finished, you can find your model under `${MODELS_DIR}/${MODEL_NAME}/`.

**Note.** Training on a single NVIDIA V100 GPU with 32GB of GPU memory requires around 4 to 5 hours of processing (training and testing). With GPUs of 12GB or less, the program will likely complain that memory has run out. As such, consider running this program in supercomputing environments.

## Test the trained-and-validated transformer

Still using `$MY_ENV`, simply call within the project's root directory the command

```sh
(set -a $MY_ENV && source $MY_ENV && ./shell-scripts/run-model/test.sh)
```

By default, testing results are written into `${MODELS_DIR}/${MODEL_NAME}-test/`.
