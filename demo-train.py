from jarvis.config import Config, from_cli
from irc.manager import IRCManager

config = Config({
    'env_param': None, 'seed': None, 'num_epochs': 12,
    'defaults': 'defaults/single_box.yaml',
    'device': 'cpu',
})

if __name__=='__main__':
    config.update(from_cli())
    env_param = config.pop('env_param')
    seed = config.pop('seed')
    num_epochs = config.pop('num_epochs')

    manager = IRCManager(**config)
    manager.train_agent(env_param, seed, num_epochs=num_epochs)
