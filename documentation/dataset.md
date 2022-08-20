# Download the Dataset Used in this Project

This project begins by deriving a shorter, Dutch variant of the Large-Scale Complex Question-Answering Dataset 2.0 (LC-QuAD 2.0). LC-QuAD 2.0 was introduced by <a href="https://doi.org/10.1007/978-3-030-30796-7_5">Dubey, Banerjee, Abdelkawi and Lehmann in 2019</a>, and can be found <a href="https://figshare.com/projects/LCQuAD_2_0/62270">on Figshare</a>.

For your convenience, however, the shell script `download.sh` may be used to automatically download the relevant data into the appropriate locations. Before executing this script, make sure you have taken the following two steps:

1. You made a copy of `.sample-env`, which you named `.env`, and entered the relevant information in said `.env` file.
2. You made sure you have `wget` at your disposal. Check this by seeing whether `wget --version` returns something sensible.

If both steps have been taken, execute the following two commands in a terminal:

```sh
# allow shell scripts to execute
chmod +x shell-scripts/create-dataset/*.sh
chmod +x shell-scripts/run-model/*.sh

(set -a .env && source .env && ./shell-scripts/create-dataset/download.sh)
```

**Note.** Is `wget` not installed? Download it on your platform. On Ubuntu 22.04 LTS, execute `sudo apt install wget` in a terminal. On macOS, using <a href="https://brew.sh">the Homebrew package manager</a>, execute `brew install wget` in a terminal.

