# Installing Python Dependencies for Running Transformer Models

Before running any of the code in `source/py-model/`, you need to have installed some dependencies. This document explains how you install them in 5 steps.

## Step 1: Ensure `python3` and `pip3` exist on your system

Python 3 and Pip 3 are both needed later on. Use your favorite package manager if either `python3 --version` or `pip3 --version` does not return with a version number. For example, Ubuntu uses `apt`, whereas macOS uses `brew`.

## Step 2: Install Miniconda

See the [regular instructions for installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) for install instructions per OS. We'll follow the one for Linux.

Execute the following commands. Once you run `bash install-miniconda.sh`, respond to the questions asked on screen. The defaults should be OK.

**Important.** If you can, ensure that `$PYTHONPATH` is empty, or, in other words: not set.

```sh
# Download the latest Miniconda installer.
cd ~/Downloads/
curl -X GET https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh --output install-miniconda.sh
chmod +x install-miniconda.sh

# Run the installer.
bash install-miniconda.sh
```

Then, quit and re-open your terminal. Changes made by Miniconda will be in effect now.

To verify that your installation works, call `conda list`; it should provide a small list of pre-provided packages present by default in Miniconda.

Finally and optionally, update your Conda installation:

```sh
conda update -n base -c defaults conda  # press 'y' if needed
```

## Step 3: Create and activate a `source/py-model/` Conda environment

Move to the project's `source/py-model/` subdirectory. Then create a Conda environment by calling

```sh
conda create --prefix .conda-env
```

This will create an empty Conda environment at `source/py-model/.conda-env`, relative to the project's root directory.

One step remains: activate your just-created Conda environment by executing

```sh
conda activate $PWD/.conda-env
```

**Tip 1.** It's a good idea to deactivate your Conda environment as soon as you start to work on another project. Call `conda deactivate` to quit the environment.

**Tip 2** To activate your Conda environment regardless of your position in the file system, fully qualify the location of `.conda-env`: `conda activate $PROJECT_ROOT/source/py-model/.conda-env/`, where `$PROJECT_ROOT` is the file system location of the project's root directory.

## Step 4: Build PyTorch from source

While still having your Conda environment activated, change directory to `source/` by calling `cd ..`. Then follow the ["build PyTorch from source" guide](https://github.com/pytorch/pytorch#from-source) on the official PyTorch GitHub repository. It is strongly recommended that you build with CUDA support enabled.

**Tip 1.** Pay attention to the prerequisites: among others, you will need a (relatively) up-to-date Python interpreter, a CUDA compiler (likely `nvcc`), among others.

**Tip 2.** Just before installing PyTorch within your Conda environment, the documentation recommends to `export CMAKE_PREFIX_PATH`. Besides doing so, also consider setting other useful environment variables. For example, it may help to `export MAX_JOBS=1` (or some other small number) if you experience crashes during the building process.

**Tip 3.** Is the build failing? Perhaps the latest commit somehow causes problems. Revert, instead, to the latest official release, and not a 'nightly' version. Do so by (1) finding the latest official release's commit hash (e.g. commit hash `67ece03` for PyTorch 1.12), (2) reverting to that commit using `git reset --hard $COMMIT_HASH`, and (3) updating all Git submodules using `git submodule update --init --recursive --jobs 0`. Then try to build from scratch. (So, remove the `pytorch/build` folder.)

**Tip 4.** Depending on the processing power of your CPU, building PyTorch from source may take one or more hours. The upshot is twofold: (1) this PyTorch build is targeted specially to your PC configuration, which may help speed up certain computations, and (2) NVIDIA Apex can be installed more easily; see the next step.

**Tip 5.** Once the installation has finished, test out PyTorch by opening a Python REPL (`python3`) and calling `import torch`. Does `torch.cuda.is_available()` work? Does a call to `torch.cuda.device_count()` yield the same number of GPUs you have in your PC? Can you create a simple tensor via `torch.tensor([[1, 2], [3, 4]], dtype=torch.float)`? It may be that PyTorch complains about actually wanting `python3 setup.py develop` instead of `python3 setup.py install`; if so, quit the REPL and call that command.

## Step 5: Build NVIDIA Apex from source

Within the same `source/` directory, follow the ["build NVIDIA Apex from source" guide](https://github.com/NVIDIA/apex#from-source) on the official NVIDIA Apex GitHub repository.

**Tip 1.** Just like tip 2 from step 4, consider calling `export MAX_JOBS=1` (or similar) before performing the install. The authors' experience is that NVIDIA Apex can at times crash during the building process unless the `MAX_JOBS` environment variable is set to a low value.

**Tip 2.** NVIDIA Apex shouldn't take as long to build as PyTorch.

**Tip 3.** Again, test your installation: enter a REPL, and call `import apex` and `ddp = apex.parallel.DistributedDataParallel`. If both work, all should be OK.

