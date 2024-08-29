# Artificial Speech Detection Framework (ASDF)

ASDF is a framework for training anti-spoofing neural networks for automatic speaker verification.

## Main features
- Multi-GPU training with DistributedDataParallel
- Multi-GPU scoring of audio files of variable lengths


## Installation
1) Clone ASDF repository
   1) Navigate to a folder, where you want ASDF to be placed to
   2) `git clone https://github.com/vvestman/asdf.git`
   3) `cd asdf`

Use an environment such as the following:
1) `conda create -n asdf python=3.12`
2) `conda activate asdf`
3) `pip install -r requirements.txt`


## Running the Wav2Vec-AASIST recipe
- See instructions from [asdf/recipes/wav2vec_aasist/README.md](asdf/recipes/wav2vec_aasist/README.md)


## Other instructions
- For more information on how to execute and configure experiments, see [asdf/src/settings/README.md](asdf/src/settings/README.md)
- To train neural networks by using multiple GPUs in parallel, see [multigpu_readme.md](multigpu_readme.md)
- To create custom network architectures, see [custom_architectures_readme.md](custom_architectures_readme.md)
 