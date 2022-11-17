import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

config = Config({
    'data_path': None,
    'env_param_grid': 'param_grids/indie_boxes.yaml',
    'agent_seeds': None,
    'belief_seeds': None,
    'min_epoch': 10,
    'min_optimality': 0.95,
    'count': 1,
    'patience': 1.,
    'defaults': 'defaults/indie_boxes.yaml',
})

if __name__=='__main__':
    config.update(from_cli())
    data_path = config.pop('data_path')
    env_param_grid = config.pop('env_param_grid')
    if isinstance(env_param_grid, str):
        with open(env_param_grid) as f:
            env_param_grid = yaml.safe_load(f)
    agent_seeds = config.pop('agent_seeds')
    belief_seeds = config.pop('belief_seeds')
    min_epoch = config.pop('min_epoch')
    min_optimality = config.pop('min_optimality')

    count = config.pop('count')
    patience = config.pop('patience')

    manager = IRCManager(**config)
    manager.compute_likelihoods(
        data_path=data_path, env_param_grid=env_param_grid,
        agent_seeds=agent_seeds, belief_seeds=belief_seeds,
        min_epoch=min_epoch, min_optimality=min_optimality,
        count=count, patience=patience,
    )

