from jarvis.config import Config, from_cli
from irc.manager import AgentManager

cli_args = Config({
    'store_dir': 'irc_store',
    'defaults': 'irc_defaults/identical_boxes.yaml',
    'env_param': None,
})

if __name__=='__main__':
    cli_args.update(from_cli())
    manager = AgentManager(
        store_dir=cli_args.pop('store_dir'),
        defaults=cli_args.pop('defaults'),
    )
    manager.train_agent(**cli_args)
