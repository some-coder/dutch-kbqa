# Create the derived Dutch dataset from LC-QuAD 2.0

The creation of the dataset consists of five steps:

1. Translate the original LC-QuAD 2.0 dataset into Dutch.
2. Replace various symbols.

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
(set -a .env && source .env && ./shell-scripts/translate.sh)
```

Once this has completed, change line 8 of `shell-scripts/translate.sh` from `  --split "train"` to `  --split "test"`, and execute the command above once more. This will generate translations for the test split.

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

### Step 2.2: Post-process the dataset

Perform each of the following for both the 'train' and 'test dataset splits of LC-QuAD 2.0. You can switch between them by changing the `$SPLIT` environment variable in your `.env` file, and re-running each of the shell scripts, in order, as described in the following.

First, replace various special symbols:

```sh
(set -a .env && source .env && ./shell-scripts/replace-special-symbols.sh)
```

Second, replace `ERROR`s by proper references:

```sh
(set -a .env && source .env && ./shell-scripts/replace-errors.sh)
```

Third, per question, collect the WikiData entities and properties present in said questions:

```sh
(set -a .env && source .env && ./shell-scripts/generate-question-entities-properties-map.sh)
```

Fourth, for each unique WikiData entity and property, generate labels for these entities and properties. Run this step one time with the environment variable `$LABEL_LANGUAGE` set to `$SOURCE_LANGUAGE`, and one time with `$LABEL_LANGUAGE` set to `$TARGET_LANGUAGE`:

```sh
# First run: In your `.env`, set `$LABEL_LANGUAGE` to `$SOURCE_LANGUAGE`. Then run the line below:
(set -a .env && source .env && ./shell-scripts/label-entities-and-properties.sh)

# Second run: Now set it to `$TARGET_LANGUAGE`. Then run the line below:
(set -a .env && source .env && ./shell-scripts/label-entities-and-properties.sh)
```

(More information will follow.)

