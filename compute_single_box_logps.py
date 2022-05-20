import argparse, json, random, time
from itertools import product
from irc.agents import BeliefAgentFamily
from boxforage.single_box import SingleBoxForaging

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='cache')
    parser.add_argument('--envs-spec-path', default='jsons/single_box_envs.json')
    parser.add_argument('--episode-path', default='cache/episodes/single_box_example.pickle')
    parser.add_argument('--max-seed', default=6, type=int)
    parser.add_argument('--min-num-epochs', default=40, type=int)
    parser.add_argument('--num-repeats', default=8, type=int)
    parser.add_argument('--max-wait', default=1., type=float)
    parser.add_argument('--num-works', default=1, type=int)
    parser.add_argument('--patience', default=168., type=float)
    args = parser.parse_args()

    time.sleep(random.random()*args.max_wait)
    bafam = BeliefAgentFamily(SingleBoxForaging, store_dir=args.store_dir)

    with open(args.envs_spec_path, 'r') as f:
        envs_spec = json.load(f)
    env_params = list(product(
        envs_spec['p_appear'], envs_spec['p_cue'], envs_spec['r_food'],
    ))
    random.shuffle(env_params)

    bafam.compute_logps(
        env_params, episode_path=args.episode_path,
        seeds=range(args.max_seed),
        min_num_epochs=args.min_num_epochs,
        num_repeats=args.num_repeats,
        num_works=args.num_works,
        patience=args.patience,
        verbose=1,
    )
