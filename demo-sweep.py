import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

config = Config({
    'env_param_grid': 'param_grids/single_box.yaml',
    'seeds': None,
    'num_epochs': 12,
    'count': 6,
    'defaults': 'defaults/single_box.yaml',
    'device': 'cpu',
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

    manager = IRCManager(**config)
    manager.train_agents(
        env_param_grid=env_param_grid, seeds=seeds,
        num_epochs=num_epochs, count=count,
    )