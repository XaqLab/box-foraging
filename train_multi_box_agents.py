import argparse, json, random
from itertools import product
from irc.agents import BeliefAgentFamily
from boxforage.multi_box import IdenticalBoxForaging

parser = argparse.ArgumentParser()
parser.add_argument('--num-boxes', default=2, type=int)
parser.add_argument('--eval-interval', default=5, type=int)
parser.add_argument('--save-interval', default=5, type=int)
parser.add_argument('--envs-spec-path', default='jsons/multi_box_envs.json')
parser.add_argument('--est-spec-path', default='jsons/two_box_est.json')
parser.add_argument('--max-seed', default=6, type=int)
parser.add_argument('--num-epochs', default=40, type=int)
parser.add_argument('--num-works', default=1, type=int)
args = parser.parse_args()

if __name__=='__main__':
    env_spec = {'boxes': {'num_boxes': args.num_boxes}}
    with open(args.est_spec_path, 'r') as f:
        est_spec = json.load(f)

    bafam = BeliefAgentFamily(
        IdenticalBoxForaging,
        env_kwargs={'env_spec': env_spec},
        model_kwargs={'est_spec': est_spec},
        state_dist_kwargs={'idxs': [[i] for i in range(args.num_boxes+1)]},
        obs_dist_kwargs={'idxs': [[i] for i in range(args.num_boxes+1)]},
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )

    with open(args.envs_spec_path, 'r') as f:
        envs_spec = json.load(f)
    env_params = list(product(
        envs_spec['p_appear'], envs_spec['p_vanish'],
        envs_spec['p_true'], envs_spec['p_false'],
    ))
    random.shuffle(env_params)
    param_tail = (10.,)+(-1.,)*args.num_boxes+(-0.5,)
    env_params = [(*env_param, *param_tail) for env_param in env_params]

    bafam.train_agents(
        env_params, seeds=range(args.max_seed),
        num_epochs=args.num_epochs,
        num_works=args.num_works, verbose=1,
    )
