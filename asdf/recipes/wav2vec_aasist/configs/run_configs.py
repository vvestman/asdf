# pylint: skip-file

sad-off
recipe.start_stage = 0
recipe.end_stage = 0
paths.sad_folder = 'sad-off'
sad.mode = 'off'
recipe.evaluation_sets = ('asvspoof21-df-progress', 'asvspoof21-la-progress', 'asvspoof21-df-eval', 'asvspoof21-la-eval')

sad-endpoint < sad-off
sad.mode = 'endpoint'
paths.sad_folder = 'sad-endpoint'

sad-on < sad-off
sad.mode = 'on'
paths.sad_folder = 'sad-on'



# Default training settings to inherit
train
network.network_class = 'asdf.src.networks.architectures2.DefaultNetwork'
recipe.start_stage = 1
recipe.end_stage = 1
network.print_interval = 5
network.scoring_print_interval = 10
network.target_loss = 0
network.min_loss_change_ratio = 0.001
network.max_epochs = 50
network.weight_decay = 0.0003
network.initial_learning_rate = 0.001
network.max_consecutive_lr_updates = 3
network.lr_update_ratio = 0.5
network.epochs_per_train_call = 10
network.optimizer = 'adam'
#recipe.training_sets = ('asvspoof19', 'wavefake', 'for', 'itw', 'mlaad', 'm-ailabs')
recipe.training_sets = ('asvspoof19',)
recipe.evaluation_sets = ('asvspoof19-dev',)
network.eval_minibatch_size_factor = 1
network.min_evaluation_utterance_length_in_samples = 16000
network.max_evaluation_utterance_length_in_samples = 1000000
rawboost.enabled = False
rawboost.applyingRatio = 0.5

# Train the default network without SAD
train-default-network < train < sad-off
paths.system_folder = 'default-network'
network.train_clip_size = 16000
network.minibatch_size = 16  #Adjust based on GPU memory

# Resume training from specific epoch
net-resume < train-default-network
network.resume_epoch = 9

# Evaluate specific epoch
eval-default-network < train-default-network
recipe.start_stage = 2
recipe.end_stage = 2
recipe.selected_epochs = (24, 23)
#recipe.evaluation_sets = ('asvspoof21-la-progress', 'asvspoof21-la-eval')
recipe.evaluation_sets = ('asvspoof19-eval', 'asvspoof21-la-progress')
#recipe.evaluation_sets = ('asvspoof19-dev', 'asvspoof19-eval', 'asvspoof21-la-progress', 'asvspoof21-la-eval')
