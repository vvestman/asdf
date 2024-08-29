## Wav2Vec 2.0 - AASIST antispoofing recipe

Unoptimized recipe for training the wav2vec-AASIST network.

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

- To execute all steps at once: \
    `python asdf/recipes/wav2vec_aasist/run.py sad-off train-ssl-aasist eval-ssl-aasist`





## Notes

- Training fully-fledged antispoofing systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, consider using the following (or similar) command: \
    `nohup python -u asdf/recipes/wav2vec_aasist/run.py train-ssl-aasist > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.
