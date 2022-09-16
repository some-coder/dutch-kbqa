# Dutch KBQA

This Git repository stores the code for the paper _BERT-based Transformer Fine-Tuning for Dutch Wikidata Question-Answering_, written by and presented at the 32nd Meeting of Computational Linguistics in the Netherlands (CLIN 32, 2022) by Niels de Jong and Dr. Gosse Bouma from the University of Groningen, Faculty of Arts.

## Overview

The paper's methodology splits up in two parts:

1. Generate a Dutch counterpart to LC-QuAD 2.0 (also see the _Thanks_ section below) with some additional post-processing.
2. Fine-tune, validate, and test various configurations for pre-trained BERT-based Transformer models.

The first step is implemented in the subdirectories `source/py-ds-create` and `source/cpp-ds-create`, whereas the second step is implemented in subdirectory `source/py-model`.

We have prepared shell scripts that take care of calling the right programs with the right parameters for you. These scripts can be found under `shell-scripts/create-dataset/` and `shell-scripts/run-model/`. However, the scripts rely on setting certain environment variables; these need to be set before you can run the scripts. The environment variables can be set jointly using 'dot-env files' placed under `environments/original/` (for the original experiments) and `environments/appendix/` (for the experiments of the appendix). Alternatively, you can create your own environment from the stub environment `.sample-env`.

In order to perform part 1 of the methodology (generating a dataset), please read the following files under `documentation/`, in the provided order:

1. `documentation/google-cloud.md`.
1. `documentation/download-dataset.md`.
2. `documentation/create-dutch-dataset.md`.

In order to perform part 2 of the methodology (developing BERT-based Transformer models), please read the following files under `documentation/`, again in the provided order:

1. `documentation/install-model-dependencies.md`.
2. `documentation/run-model.md`.

If you run into any obstacles, please do not hesitate to contact us; see the _Contact_ section below.

## Thanks

This project would not have been possible without:

1. Dubey, Banerjee, Abdelkawi, and Lehmann (from _LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia_, 2021, <a href="https://doi.org/10.1007/978-3-030-30796-7_5">link</a>), who provide the LC-QuAD 2.0 dataset for part 1 of our methodology.
2. Cui, Aralikatte, Lent, and Herschovich (from _Multilingual Compositional Wikidata Questions_, 2021, <a href="https://arxiv.org/abs/2108.03509v1">link</a>), on which we base large parts of our dataset-generating methodology.
3. Tran, Phan, Anibal, Nguyen, and Nguyen (from _SPBERT: an Efficient Pre-training BERT on SPARQL Queries for Question Answering over Knowledge Graphs_, 2021, <a href="https://doi.org/10.1007/978-3-030-92185-9_42">link</a>), whose <a href="https://github.com/heraclex12/NLP2SPARQL">GitHub repository</a> formed the basis for our methodology's part 2.
4. The support of the maintainers of the University of Groningen's <a href="https://www.rug.nl/society-business/centre-for-information-technology/research/services/hpc/facilities/peregrine-hpc-cluster?lang=en">Peregrine supercomputing cluster</a>.

## Contact

Please contact <a href="mailto:najong99@hotmail.nl">Niels de Jong via email</a>, or open an issue in the _Issues_ tab of this repository.

