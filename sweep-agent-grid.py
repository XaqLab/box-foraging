import yaml
from jarvis.config import Config, from_cli
from irc.manager import IRCManager

cli_args = Config({
    'store_dir': 'irc_store',
    'agent_defaults': 'irc_defaults/identical_boxes.yaml',
    'env_param_grid': 'param_grids/identical_boxes.yaml',
    'seeds': None,
})

if __name__=='__main__':
    cli_args.update(from_cli())
    manager = IRCManager(
        store_dir=cli_args.pop('store_dir'),
        agent_defaults=cli_args.pop('agent_defaults'),
    )
    manager.train_agents(**cli_args)
