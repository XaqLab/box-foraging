import argparse, json, random, pickle, time
from itertools import product
from irc.agents import BeliefAgentFamily
from boxforage.single_box import SingleBoxForaging
from jarvis.utils import tensor_dict, time_str
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--envs-spec-path', default='jsons/single_box_envs.json')
parser.add_argument('--episode-path', default='cache/episodes/single_box_example.pickle')
parser.add_argument('--num-repeats', default=8, type=int)
parser.add_argument('--save-path', default='cache/logps/single_box_example.pickle')
args = parser.parse_args()

if __name__=='__main__':
    bafam = BeliefAgentFamily(
        SingleBoxForaging,
    )

    with open(args.envs_spec_path, 'r') as f:
        envs_spec = json.load(f)
    env_params = list(product(
        envs_spec['p_appear'], envs_spec['p_cue'], envs_spec['r_food'],
    ))
    random.shuffle(env_params)

    with open(args.episode_path, 'rb') as f:
        episode = pickle.load(f)['episode']
    actions = episode['actions']
    obss = episode['obss']

    logps = {}
    for env_param in env_params:
        with open(args.save_path, 'rb') as f:
            logps = pickle.load(f)['logps']
        if env_param in logps:
            continue
        print(f"Calculating log likelihood for {env_param}...")
        tic = time.time()
        logps[env_param] = []
        for key in bafam.completed(cond={'env_param': env_param}):
            config = bafam.configs[key]
            _, ckpt = bafam.load_ckpt(config)
            agent = bafam.create_agent(config)
            agent.load_state_dict(tensor_dict(ckpt['agent_state']))
            _logps = agent.episode_likelihood(actions, obss, num_repeats=args.num_repeats)
            logps[env_param].append(_logps)
        logps[env_param] = np.array(logps[env_param])
        with open(args.save_path, 'wb') as f:
            pickle.dump({
                'actions': actions, 'obss': obss,
                'logps': logps,
            }, f)
        toc = time.time()
        print("{} agents analyzed ({})".format(len(logps[env_param]), time_str(toc-tic)))
