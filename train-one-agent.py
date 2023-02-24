from jarvis.config import Config, from_cli
from irc.manager import IRCManager

args = Config({
    'store_dir': 'irc_store',
    'agent_defaults': 'irc_defaults/identical_boxes.yaml',
    'env_param': None,
    'seed': None,
    'num_epochs': 30,
})

if __name__=='__main__':
    args.update(from_cli())
    manager = IRCManager(
        store_dir=args.store_dir,
        agent_defaults=args.agent_defaults,
    )
    manager.train_agent(
        args.env_param, args.seed, args.num_epochs,
    )
