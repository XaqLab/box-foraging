import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_single_box_episode(agent, env=None, episode=None, num_steps=40, figsize=(10, 1.5)):
    if episode is None:
        episode = agent.run_one_episode(env, num_steps)
    num_steps = episode['num_steps']
    states = episode['states']
    obss = episode['obss']
    actions = episode['actions']
    rewards = episode['rewards']
    beliefs = episode['beliefs']

    fig_w, fig_h = figsize
    aspect = num_steps*fig_h/fig_w*1.5
    figs = []

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        states.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
        )
    cbar = plt.colorbar(h, label='Has food')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['F', 'T'])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    figs.append(fig)

    num_shades = agent.model.env.env_spec['box']['num_shades']
    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        obss.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=num_shades, origin='lower', cmap='coolwarm',
        )
    cbar = plt.colorbar(h, label='Color cue')
    cbar.set_ticks([0, num_shades])
    idxs = (actions==1)&(rewards>0)
    h_true = ax.scatter(np.arange(num_steps)[idxs], np.zeros(sum(idxs)), color='magenta', marker='o', s=50)
    idxs = (actions==1)&(rewards<0)
    h_false = ax.scatter(np.arange(num_steps)[idxs], np.zeros(sum(idxs)), color='salmon', marker='x', s=50)
    ax.legend([h_true, h_false], ['Has food', 'No food'], loc='upper left', fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    figs.append(fig)

    probs = np.zeros((num_steps+1, 1))
    for t, belief in enumerate(beliefs):
        _states = np.array([(1,)])
        probs[t] = agent.query_probs(belief, _states)[0]
    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        probs.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
        )
    cbar = plt.colorbar(h, label='Belief')
    cbar.set_ticks([0, 1])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    figs.append(fig)
    return episode, figs

def plot_multi_box_episode(agent, env=None, episode=None, num_steps=40, figsize=(10, 1.5)):
    if episode is None:
        episode = agent.run_one_episode(env, num_steps)
    episode = agent.run_one_episode(env, num_steps)
    num_boxes = agent.model.env.num_boxes
    num_steps = episode['num_steps']
    states = episode['states']
    obss = episode['obss']
    actions = episode['actions']
    rewards = episode['rewards']
    beliefs = episode['beliefs']

    fig_w, fig_h = figsize
    figs = []

    aspect = num_steps/num_boxes*fig_h/fig_w*1.5
    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        states[:, :num_boxes].T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
        )
    cbar = plt.colorbar(h, label='Has food')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['F', 'T'])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks(range(num_boxes))
    ax.set_yticklabels([f'Box {i}' for i in range(num_boxes)])
    ax.set_xlabel('Time')
    figs.append(fig)

    num_shades = agent.model.env.env_spec['boxes']['num_shades']
    aspect = num_steps/(num_boxes+1)*fig_h/fig_w*1.5
    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        obss[:, :num_boxes].T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=num_shades, origin='lower', cmap='coolwarm',
        )
    cbar = plt.colorbar(h, label='Color cue')
    cbar.set_ticks([0, num_shades])
    idxs, = np.nonzero((actions!=num_boxes+1)|(obss[:-1, -1]==num_boxes))
    t = np.arange(num_steps)
    ax.scatter(t[idxs], obss[idxs, -1], color='blue', marker='.', s=10)
    idxs, = np.nonzero((actions==num_boxes+1)&(obss[:-1, -1]<num_boxes)&(rewards>0))
    h_true = ax.scatter(t[idxs], obss[idxs, -1], color='magenta', marker='o', s=50)
    idxs, = np.nonzero((actions==num_boxes+1)&(obss[:-1, -1]<num_boxes)&(rewards<0))
    h_false = ax.scatter(t[idxs], obss[idxs, -1], color='salmon', marker='x', s=50)
    ax.legend([h_true, h_false], ['Has food', 'No food'], loc='upper left', fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_ylim([-0.5, num_boxes+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks(range(num_boxes+1))
    ax.set_yticklabels([f'Box {i}' for i in range(num_boxes)]+['Center'])
    ax.set_xlabel('Time')
    figs.append(fig)

    has_foods = np.stack(np.unravel_index(np.arange(2**num_boxes), [2]*num_boxes)).T
    probs = np.zeros((num_steps+1, num_boxes))
    for t, belief in enumerate(beliefs):
        agent_loc = states[t, -1]
        _states = np.concatenate([has_foods, np.ones((len(has_foods), 1), dtype=int)*agent_loc], axis=1)
        _prob = agent.query_probs(belief, _states)
        for b_idx in range(num_boxes):
            probs[t, b_idx] += _prob[has_foods[:, b_idx]==1].sum()
    aspect = num_steps/num_boxes*fig_h/fig_w*1.5
    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        probs.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
        )
    cbar = plt.colorbar(h, label='Belief')
    cbar.set_ticks([0, 1])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks(range(num_boxes))
    ax.set_yticklabels([f'Box {i}' for i in range(num_boxes)])
    ax.set_xlabel('Time')
    figs.append(fig)
    return episode, figs
