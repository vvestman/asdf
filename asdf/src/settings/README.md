# Managing settings

ASDF has three layers of settings:

1) Default settings
2) Recipe-wise initial configs
3) Recipe-wise run configs

- Settings defined in 2. will override settings in 1.
- Settings defined in 3. will override settings in 1. and 2.

## Default settings

Default settings are defined in [asdf/src/settings/settings.py](asdf/src/settings/settings.py) (take a look).

- Do not modify these settings unless you want to change the default settings for all systems.
- All settings can be changed in recipe-wise config files.
- `Settings` class defined in the above file follows the [singleton](https://en.wikipedia.org/wiki/Singleton_pattern) design pattern. You can access all the settings through this singleton. For example, the minibatch size can be obtained by calling \
    `minibatch_size = Settings().network.minibatch_size`
- The following recipe-wise configs override the default settings defined in [asdf/src/settings/settings.py](asdf/src/settings/settings.py).

## Recipe-wise initial configs

- Initial config file of a recipe defines settings that will be used for all executions of the recipe (unless overriden by the run configs).

- An example of initial config: [asdf/recipes/wav2vec_aasist/configs/init_config.py](aasdf/recipes/wav2vec_aasist/configs/init_config.py)

- Despite the file extension, the initial config files are not python files, but text files with custom format. The `.py` extension is used to get nice color highlighting in code editors.

- Format of the initial config file:
  - Basic Python syntax is used for assigning values for different settings.
  - Can contain empty lines (does not affect the outcome).
  - Any setting that is defined in the default settings can be updated in the initial config file.

- Where the initial config file is used?
  - In the beginning of each recipe, `Settings()` are initialized using the initial config file. For example: \
      `Settings(os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'init_config.py'))` \
      (where `fileutils.get_folder_of_file(__file__)` gives the folder of the recipe)

## Recipe-wise run configs

- An example of recipe-wise run configs: [asdf/recipes/wav2vec_aasist/configs/run_configs.py](asdf/recipes/wav2vec_aasist/configs/run_configs.py)
- A run config is a group of settings that will be used at the same time to conduct a specific task.
- Different run configs are separated using empty lines in a run config file.
- The first line of each run config defines a name for the run config.
- Run config is executed by giving the name of the config as a command line argument. For example: \
    `python asdf/recipes/wav2vec_aasist/run.py train-ssl-aasist` \
    The above trains the default network (here `train-ssl-aasist` is the name of run config defined in [asdf/recipes/wav2vec_aasist/configs/run_configs.py](asdf/recipes/wav2vec_aasist/configs/run_configs.py)
- A run config can inherit settings from one or more run configs by using `<` operator. For example: \
    `train-ssl-aasist < train < sad-off` \
    Here, the settings from run condig `sad-off` are applied first, then the settings from `train` are applied (possibly overriding the settings in `sad-off`), and finally the settings under the `train-ssl-aasist` definition are applied (overriding everything else that is set before).
- When multiple run configs are executed one after another (for example `python asdf/recipes/wav2vec_aasist/run.py sad-off train-ssl-aasist eval-ssl-aasist`), then after running each run config, the settings are returned to the initial state. Thus, the settings used in `sad-off` do not interfere when running `train-ssl-aasist` and `eval-ssl-aasist`.
- Where is the run config file applied?
  - The recipes have a for loop that looks something like this: \
  `for settings_string in Settings().load_settings(run_config_file, run_configs):` \
  This loop will iterate through all run configs that are given as a command line arguments.
