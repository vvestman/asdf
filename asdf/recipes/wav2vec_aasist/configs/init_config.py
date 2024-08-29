# pylint: skip-file

computing.network_dataloader_workers = 10
computing.use_gpu = True
computing.gpu_ids = (0, )

# Don't apply weight decay to batch norm
network.weight_decay_skiplist = ('bn',)

# TODO update this
#paths.output_folder = '/data/vvestman/asdf_outputs'
paths.output_folder = '/scratch/project_2006687/vvestman/asdf_outputs'
