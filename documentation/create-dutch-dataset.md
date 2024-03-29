# Create the derived Dutch dataset from LC-QuAD 2.0

The creation of the dataset consists of two steps:

1. Translate the original LC-QuAD 2.0 dataset into Dutch.
2. Post-process the dataset.

Perform the two steps in order as explained below.

## Step 1: Translate LC-QuAD 2.0 into Dutch

Before translating LC-QuAD 2.0, we must first prepare our Python environment. That's done by executing the following commands:

```sh
cd source/py-ds-create/
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

Once done, return to the project's root directory. (On UNIX machines, `cd ../..` should suffice, for example.) Then, translate the training dataset split of LC-QuAD 2.0 to Dutch by executing

```sh
(set -a .env && source .env && ./shell-scripts/create-dataset/translate.sh)
```

Once this has completed, the `$SPLIT` environment variable of your `.env` from `"train"` to `"test"`, and execute the command above once more. This will generate translations for the test split.

**Note.** This step assumes you already have <a href="https://www.python.org/">Python</a> and <a href="https://docs.python.org/3/library/venv.html">`venv`</a> at your disposal. If not, refer to the links to see how to obtain them for your system.

## Step 2: Replace symbols in the Dutch dataset

Next, you need to replace various special symbols found in the translated data, such as <a href="https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references">HTML character entities</a>. To achieve this, you need to make two steps:

1. Build the C++ project for post-processing the Dutch LC-QuAD 2.0 derivative.
2. Post-process the dataset using said C++ project.

### Step 2.1: Build the C++ post-processing project

Begin by moving to the `cpp-ds-create` subdirectory, and installing the <a href="https://vcpkg.io/en/index.html">`vcpkg`</a> package manager for C++:

```sh
cd source/cpp-ds-create/
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
```

Then install the following dependencies:

```sh
./vcpkg/vcpkg install curlpp
./vcpkg/vcpkg install jsoncpp
./vcpkg/vcpkg install utfcpp
./vcpkg/vcpkg install boost-program-options
```

Finally, build the project using <a href="https://cmake.org">CMake</a>:

```sh
mkdir build/
cmake -B build/ \
      -S . \
      -DCMAKE_TOOLCHAIN_FILE="${PWD}/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build/  # this may take a while
```

**Note.** In the first `cmake` call above, you can optionally add the argument `-G "Ninja"` to potentially speed up the build process. Thi does, however, require that you have `ninja` installed on your system. On Ubuntu and other Linux distributions, something like `sudo apt install ninja-build` should suffice; on Macs, use `brew install ninja`.

### Step 2.2: Post-process the dataset

Perform the following 6 steps in order for both the `"train"` and `"test"` dataset splits of LC-QuAD 2.0. You do so by first executing the steps below with your `.env`'s `$SPLIT` environment variable set to `"train"`; then, you repeat the steps below once more, but now with `$SPLIT` set to `"test"`.

**Step 1.** Replace various special symbols:

```sh
(set -a .env && source .env && ./shell-scripts/create-dataset/replace-special-symbols.sh)
```

**Step 2.** Replace `ERROR`s by proper references:

```sh
(set -a .env && source .env && ./shell-scripts/create-dataset/replace-errors.sh)
```

**Step 3.** Per question, collect the WikiData entities and properties present in said questions:

```sh
(set -a .env && source .env && ./shell-scripts/create-dataset/generate-question-entities-properties-map.sh)
```

**Step 4.** For each unique WikiData entity and property, generate labels for these entities and properties. Run this step one time with the environment variable `$LABEL_LANGUAGE` set to `$SOURCE_LANGUAGE`, and one time with `$LABEL_LANGUAGE` set to `$TARGET_LANGUAGE`:

```sh
# First run: In your `.env`, set `$LABEL_LANGUAGE` to `$SOURCE_LANGUAGE`. Then run the line below:
(set -a .env && source .env && ./shell-scripts/create-dataset/label-entities-and-properties.sh)

# Second run: Now set it to `$TARGET_LANGUAGE`. Then run the line below:
(set -a .env && source .env && ./shell-scripts/create-dataset/label-entities-and-properties.sh)
```

**Step 5.** 'Mask' entities and properties in the question-answer pairs:

```sh
(set -a .env && source .env && ./shell-scripts/create-dataset/mask-question-answer-pairs.sh)
```

**Step 6.** Call the finalisation operation:

```sh
(set -a .env && source .env && ./shell-scripts/create-dataset/finalise-dataset.sh)
```

If you have performed steps 1 up until 6 for both the `"train"` and `"test"` dataset splits of LC-QuAD 2.0, you have successfully created a derived counterpart to LC-QuAD 2.0; it can now be used for model training.

