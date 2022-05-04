import random
from irc.agents import BeliefAgentFamily
from boxforage.single_box import SingleBoxForaging

if __name__=='__main__':
    bafam = BeliefAgentFamily(
        SingleBoxForaging,
        save_interval=5,
        s_pause=2., l_pause=10.,
    )

    env_params = []
    for p_appear in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for p_cue in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for r_food in [2, 5, 10]:
                env_params.append((p_appear, p_cue, r_food))
    random.shuffle(env_params)
    bafam.train_agents(env_params, num_epochs=30, verbose=1)
