## Wav2Vec 2.0 - AASIST antispoofing recipe

Unoptimized barebones recipe demonstrating training of wav2vec-AASIST network.

## Requirements
- ASVSpoof 19 dataset

## Recipe preparation

Update the settings in the initial config file [configs/init_config.py](configs/init_config.py):

- The computing settings below can be changed based on your system:
  
```txt
    computing.network_dataloader_workers = 10
    computing.use_gpu = True
    computing.gpu_ids = (0,)
```

- Change `paths.output_folder` to point to the desired output folder where all the outputs (networks, scores, results, etc...) should be stored. The folder does not need to exist before running the scripts.
  - Typically you would want to have different output folder for each recipe to avoid accidentally overridding outputs of other recipes.
- Change ASVspoof19 related dataset folders in [asdf/recipes/wav2vec_aasist/run.py](asdf/recipes/wav2vec_aasist/run.py).

## Running the recipe

- Activate python environment: `conda activate asdf`

- To execute the recipe step-by-step run the following:
1) `python asdf/recipes/wav2vec_aasist/run.py sad-off`
    - This will pre-determine utterance lengths (used to achieve faster scoring). 
2) `python asdf/recipes/wav2vec_aasist/run.py train-ssl-aasist`
      - This will train the default network.
3) `python asdf/recipes/wav2vec_aasist/run.py eval-ssl-aasist`
    - This will evaluate the network at specific epoch/epochs (change the epoch from the run config file).

- To execute both steps at once: \
    `python asdf/recipes/wav2vec_aasist/run.py sad-off train-ssl-aasist eval-ssl-aasist`

## Expected results

### Results in with full PLDA training data and score normalization (scoring stage)

``` txt
EER = 2.9640  minDCF = 0.2010  [epoch 37] [vox1_original]
EER = 2.8344  minDCF = 0.1985  [epoch 37] [vox1_cleaned]
EER = 2.8438  minDCF = 0.1813  [epoch 37] [vox1_extended_original]
EER = 2.7514  minDCF = 0.1792  [epoch 37] [vox1_extended_cleaned]
EER = 4.7982  minDCF = 0.2852  [epoch 37] [vox1_hard_original]
EER = 4.6833  minDCF = 0.2834  [epoch 37] [vox1_hard_cleaned]
```

### Results during network training with limited PLDA data

``` txt
EER = 7.2271  minDCF = 0.4130  [epoch 5] [vox1_original]
EER = 7.0806  minDCF = 0.4112  [epoch 5] [vox1_cleaned]
EER = 5.6735  minDCF = 0.3254  [epoch 10] [vox1_original]
EER = 5.5517  minDCF = 0.3233  [epoch 10] [vox1_cleaned]
EER = 4.8570  minDCF = 0.2966  [epoch 15] [vox1_original]
EER = 4.6849  minDCF = 0.2945  [epoch 15] [vox1_cleaned]
EER = 4.5919  minDCF = 0.2833  [epoch 20] [vox1_original]
EER = 4.4563  minDCF = 0.2811  [epoch 20] [vox1_cleaned]
EER = 4.1783  minDCF = 0.2599  [epoch 25] [vox1_original]
EER = 4.0468  minDCF = 0.2577  [epoch 25] [vox1_cleaned]
EER = 4.1571  minDCF = 0.2464  [epoch 30] [vox1_original]
EER = 4.0521  minDCF = 0.2441  [epoch 30] [vox1_cleaned]
EER = 4.1093  minDCF = 0.2485  [epoch 35] [vox1_original]
EER = 3.9989  minDCF = 0.2461  [epoch 35] [vox1_cleaned]
EER = 4.0934  minDCF = 0.2480  [epoch 37] [vox1_original]
EER = 3.9670  minDCF = 0.2457  [epoch 37] [vox1_cleaned]
```



## Notes

- Training fully-fledged antispoofing systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, consider using the following (or similar) command: \
    `nohup python -u asdf/recipes/wav2vec_aasist/run.py train-ssl-aasist > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.
