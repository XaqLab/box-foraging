import random
from irc.agents import BeliefAgentFamily
from boxforage.multi_box import IdenticalBoxForaging

if __name__=='__main__':
    est_spec = {
        'state_prior': {
            'num_samples': 20000,
            'optim_kwargs': {
                'batch_size': 128, 'num_epochs': 40,
            },
        },
        'obs_conditional': {
            'num_samples': 20000,
            'optim_kwargs': {
                'batch_size': 128, 'num_epochs': 40,
            },
        },
        'belief': {
            'num_samples': 800,
            'optim_kwargs': {
                'batch_size': 64, 'num_epochs': 10,
            },
        },
    }

    bafam = BeliefAgentFamily(
        IdenticalBoxForaging,
        model_kwargs={'est_spec': est_spec},
        state_dist_kwargs={'idxs': [[0], [1], [2]]},
        obs_dist_kwargs={'idxs': [[0], [1], [2]]},
        eval_interval=2,
        save_interval=6,
        s_pause=2., l_pause=10.,
    )

    env_params = []
    for p_appear in [0.05, 0.08, 0.11, 0.14, 0.17, 0.2]:
        for p_vanish in [0.05, 0.1, 0.15]:
            for p_true in [0.4, 0.6, 0.8]:
                for p_false in [0.2, 0.4, 0.6]:
                    env_spec = {
                        'boxes': {
                            'p_appear': p_appear, 'p_vanish': p_vanish,
                            'p_true': p_true, 'p_false': p_false,
                        },
                    }
                    env_params.append(IdenticalBoxForaging(env_spec=env_spec).get_env_param())
    random.shuffle(env_params)
    bafam.train_agents(env_params, num_epochs=30, verbose=1)
