import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

args = Config({
    'store_dir': 'irc_store',
    'agent_defaults': 'irc_defaults/identical_boxes.yaml',
    'env_param_grid': 'param_grids/identical_boxes.yaml',
    'seeds': None,
    'num_epochs': 12,
    'count': 1,
    'patience': 4.,
})

if __name__=='__main__':
    args.update(from_cli())
    manager = IRCManager(
        store_dir=args.store_dir,
        agent_defaults=args.agent_defaults,
    )

    env_param_grid = args.env_param_grid
    if isinstance(env_param_grid, str):
        with open(env_param_grid) as f:
            print(f"Environment parameter grid loaded from '{env_param_grid}'.")
            env_param_grid = yaml.safe_load(f)
    manager.train_agents(
        env_param_grid=env_param_grid, seeds=args.seeds,
        num_epochs=args.num_epochs, count=args.count, patience=args.patience,
    )
