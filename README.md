# Artificial Speech Detection Framework (ASDF)

ASDF is a framework for training anti-spoofing neural networks for automatic speaker verification.

## Main features
- Multi-GPU training with DistributedDataParallel
- Multi-GPU scoring of audio files of variable lengths

## Requirements
- GPU with at least 10 GB of memory (More recommended)
- A computing server with 5-50 CPU cores per GPU and ample amount of RAM
- Python environment (installation instructions below)
- Anti-spoofing datasets (asvspoofXX, MLAAD, etc...)

## Installation

1) Clone ASDF repository
   1) Navigate to a folder, where you want ASDF to be placed to
   2) `git clone https://gitlab.com/ville.vestman/asdf.git`
   3) `cd asdf`

If you running systems locally or in servers, use environment such as the following:

1) `conda create -n asdf python=3.12`
2) `conda activate asdf`
3) `conda install pytorch`
4) `pip install torchaudio`
5) `pip install soundfile`
6) `conda install scipy`
7) `conda install matplotlib`
8) `pip install omegaconf`
9) `pip install hydra-core`
10) `pip install bitarray`
11) `pip install tqdm`
12) `pip install scikit-learn`
13) `pip install sacrebleu`












## Running the Wav2Vec-AASIST recipe

- See instructions from [asdf/recipes/wav2vec_aasist/README.md](asdf/recipes/wav2vec_aasist/README.md)


## Other instructions
- For more information on how to execute and configure experiments, see [asdf/src/settings/README.md](asdf/src/settings/README.md)
- To train neural networks by using multiple GPUs in parallel, see [multigpu_readme.md](multigpu_readme.md)
- To create custom network architectures, see [custom_architectures_readme.md](custom_architectures_readme.md)

## License

ASDF is licensed under the MIT license. See [LICENSE.txt](LICENSE.txt). 
 