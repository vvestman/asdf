## Using multiple GPUs in parallel to train neural networks

ASDF implements multi-GPU training and scoring using `DistributedDataParallel` and `torchrun` utilities of PyTorch.

Instead of the typical single GPU execution:
```txt
python asdf/recipes/wav2vec_aaist/run.py train-ssl-aasist
```
multi-GPU training can be started with a command:
```txt
torchrun --nproc_per_node=2 asdf/recipes/wav2vec_aaist/run.py train-ssl-aasist
```

Here `--nproc_per_node` specifies the number of GPUs. By default, the above command will use the first two GPUs that you have (or that are visible as determined by CUDA_VISIBLE_DEVICES environment variable). 

`Settings().computing.gpu_ids` allows you to select which GPUs will be used. For example, if you have four GPUs and want to use all except the second GPU, then set `computing.gpu_ids=(0,2,3)` and run
```txt
torchrun --nproc_per_node=3 asdf/recipes/wav2vec_aaist/run.py train-ssl-aasist
```
The best place to change `computing.gpu_ids` setting is in either in the `init_config.py` or `run_configs.py` file of the recipe.

* For more info on how to use torchrun: [https://pytorch.org/docs/stable/elastic/run.html](https://pytorch.org/docs/stable/elastic/run.html).

* If you are running scripts in CSC:s servers, you may use the scripts in  [puhti_slurm_scripts](puhti_slurm_scripts) folder.



## Concurrent training of multiple models

The framework also makes it easy to simultaneously train multiple different models (one model per GPU).

For example, in `run_configs.py` you may have:
```
net
paths.system_folder = 'system'
computing.gpu_ids=(0,)
recipe.start_stage = 5
recipe.end_stage = 5
...
...

larger_net < net
paths.system_folder = 'larger_system'
computing.gpu_ids = (1,)
...
...
```

And then you can execute back to back
```txt
python asdf/recipes/wav2vec_aaist/run.py net
```
and
```txt
python asdf/recipes/wav2vec_aaist/run.py larger_net
```

This will train two networks at the same time. The first network is trained using GPU 0 and will be saved to a relative folder called `system`. The second network is trained using GPU 1 and will be saved to a relative folder called `larger_system`.

`larger_net < net` means that `larger_net` will inherit settings from `net`. After that it will override some of the settings to make the networks different.
