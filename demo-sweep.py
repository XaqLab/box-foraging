import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

config = Config({
    'env_param_grid': 'param_grids/indie_boxes.yaml',
    'seeds': None,
    'num_epochs': 12,
    'count': 1,
    'patience': 1.,
    'defaults': 'defaults/indie_boxes.yaml',
})

if __name__=='__main__':
    config.update(from_cli())
    env_param_grid = config.pop('env_param_grid')
    if isinstance(env_param_grid, str):
        with open(env_param_grid) as f:
            env_param_grid = yaml.safe_load(f)
    seeds = config.pop('seeds')
    num_epochs = config.pop('num_epochs')
    count = config.pop('count')
    patience = config.pop('patience')

    manager = IRCManager(**config)
    manager.train_agents(
        env_param_grid=env_param_grid, seeds=seeds,
        num_epochs=num_epochs, count=count, patience=patience,
    )
