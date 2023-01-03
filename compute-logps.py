import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

cli_args = Config({
    'episode_path': None,
    'env_param_grid': 'param_grids/single_box.yaml',
    'agent_seeds': None,
    'belief_seeds': None,
    'min_epoch': 20,
    'min_optimality': 0.98,
    'count': 1,
    'patience': 4.,
    'disp_interval': 20,
    'defaults': 'irc_defaults/single_box.yaml',
})

if __name__=='__main__':
    cli_args.update(from_cli())
    episode_path = cli_args.pop('episode_path')
    env_param_grid = cli_args.pop('env_param_grid')
    if isinstance(env_param_grid, str):
        with open(env_param_grid) as f:
            env_param_grid = yaml.safe_load(f)
    agent_seeds = cli_args.pop('agent_seeds')
    belief_seeds = cli_args.pop('belief_seeds')
    min_epoch = cli_args.pop('min_epoch')
    min_optimality = cli_args.pop('min_optimality')

    count = cli_args.pop('count')
    patience = cli_args.pop('patience')
    disp_interval = cli_args.pop('disp_interval')

    manager = IRCManager(**cli_args)
    manager.compute_logps(
        episode_path=episode_path, env_param_grid=env_param_grid,
        agent_seeds=agent_seeds, belief_seeds=belief_seeds,
        min_epoch=min_epoch, min_optimality=min_optimality,
        count=count, patience=patience, disp_interval=disp_interval,
    )
