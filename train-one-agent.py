from jarvis.config import Config, from_cli
from irc.manager import IRCManager

cli_args = Config({
    'env_param': None, 'seed': None,
    'num_epochs': 12,
    'defaults': 'irc_defaults/single_box.yaml',
})

if __name__=='__main__':
    cli_args.update(from_cli())
    env_param = cli_args.pop('env_param')
    seed = cli_args.pop('seed')
    num_epochs = cli_args.pop('num_epochs')

    manager = IRCManager(**cli_args)
    manager.train_agent(env_param, seed, num_epochs=num_epochs)
