import yaml, pickle
from datetime import datetime
from jarvis.config import Config, from_cli
from irc.manager import AgentManager

cli_args = Config({
    'episode_path': None,
    'defaults': 'irc_defaults/identical_boxes.yaml',
    'env_param_grid': 'param_grids/identical_boxes.yaml',
    'likelihood_path': None,
})

if __name__=='__main__':
    cli_args.update(from_cli())
    with open(cli_args.pop('episode_path'), 'rb') as f:
        saved = pickle.load(f)
    observations = saved['observations']
    actions = saved['actions']
    manager = AgentManager(
        defaults=cli_args.pop('defaults'),
    )
    env_param_grid = cli_args.pop('env_param_grid')
    if isinstance(env_param_grid, str):
        with open(env_param_grid, 'r') as f:
            env_param_grid = yaml.safe_load(f)
    likelihood_path = cli_args.pop('likelihood_path')
    if likelihood_path is None:
        now = datetime.now()
        likelihood_path = 'likelihood_{}.pickle'.format(now.strftime('%H%M-%m%y'))
    counts, logps = manager.compute_logps(
        observations, actions,
        env_param_grid=env_param_grid,
        **cli_args,
    )
    with open(likelihood_path, 'wb') as f:
        pickle.dump({
            'observations': observations, 'actions': actions,
            'defaults': manager.defaults.asdict(),
            'env_param_grid': env_param_grid,
            'counts': counts, 'logps': logps, 
        }, f)
