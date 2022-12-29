import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

cli_args = Config({
    'env_param_grid': 'param_grids/single_box.yaml',
    'seeds': None,
    'num_epochs': 12,
    'count': 1,
    'patience': 4.,
    'defaults': 'irc_defaults/single_box.yaml',
})

if __name__=='__main__':
    cli_args.update(from_cli())
    env_param_grid = cli_args.pop('env_param_grid')
    if isinstance(env_param_grid, str):
        with open(env_param_grid) as f:
            print("Environment parameter grid loaded from {env_param_grid}")
            env_param_grid = yaml.safe_load(f)
    seeds = cli_args.pop('seeds')
    num_epochs = cli_args.pop('num_epochs')
    count = cli_args.pop('count')
    patience = cli_args.pop('patience')

    manager = IRCManager(**cli_args)
    manager.train_agents(
        env_param_grid=env_param_grid, seeds=seeds,
        num_epochs=num_epochs, count=count, patience=patience,
    )
