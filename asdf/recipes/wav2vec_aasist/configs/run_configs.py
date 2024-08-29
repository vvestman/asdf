# pylint: skip-file

sad-off
recipe.start_stage = 0
recipe.end_stage = 0
paths.sad_folder = 'sad-off'
sad.mode = 'off'
recipe.evaluation_sets = ('asvspoof19-dev', 'asvspoof19-eval')

sad-endpoint < sad-off
sad.mode = 'endpoint'
paths.sad_folder = 'sad-endpoint'

sad-on < sad-off
sad.mode = 'on'
paths.sad_folder = 'sad-on'



# Default training settings to inherit
train
network.network_class = 'asdf.src.networks.architectures.SSL_AASIST'
recipe.start_stage = 1
recipe.end_stage = 1
network.print_interval = 10
network.target_loss = 0
network.min_loss_change_ratio = 0.001
network.max_epochs = 50
network.weight_decay = 0.001
network.initial_learning_rate = 0.0001
network.max_consecutive_lr_updates = 3
network.lr_update_ratio = 0.5
network.epochs_per_train_call = 1
network.optimizer = 'adam'
#recipe.training_sets = ('asvspoof19', 'wavefake', 'for', 'itw', 'mlaad', 'm-ailabs')
recipe.training_sets = ('asvspoof19', )
recipe.evaluation_sets = ('asvspoof19-dev', )
network.eval_minibatch_size_factor = 1
network.min_evaluation_utterance_length_in_samples = 64000
network.max_evaluation_utterance_length_in_samples = 1000000


# Train the default network without SAD
train-ssl-aasist < train < sad-off
paths.system_folder = 'ssl-aasist'
network.train_clip_size = 64000
network.minibatch_size = 200  #Adjust based on GPU memory

# Resume training from specific epoch
net-resume < train-ssl-aasist
network.resume_epoch = 7

# Evaluate specific epoch
eval-ssl-aasist < train-ssl-aasist
recipe.start_stage = 2
recipe.end_stage = 2
recipe.selected_epochs = (17, 19)
recipe.evaluation_sets = ('asvspoof19-dev', 'asvspoof19-eval')