import matplotlib.pyplot as plt
import numpy as np
from itertools import product

def plot_single_box_episode(episode, num_shades=None, figsize=(5, 4)):
    num_steps = episode['num_steps']
    states = episode['states']
    observations = episode['observations']
    actions = episode['actions']
    rewards = episode['rewards']
    assert not np.any(episode['q_states']-np.array([(1,)]))
    probs = episode['q_probs']
    if num_shades is None:
        num_shades = observations.max()

    fig = plt.figure(figsize=figsize)
    fig_w, fig_h = figsize
    height = 0.26
    gap = (1-3*height)/4
    aspect = num_steps*fig_h/fig_w*(height/0.9)

    ax = plt.axes([0.05, 3*gap+2*height, 0.9, height])
    h = ax.imshow(
        states.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
    )
    cax = plt.axes([0.97, 3*gap+2*height, 0.03, height])
    cbar = plt.colorbar(h, cax=cax, label='Has food')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['F', 'T'])
    ax.spines['left'].set_visible(False)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_xticklabels([])
    ax.set_yticks([])

    ax = plt.axes([0.05, 2*gap+height, 0.9, height])
    h = ax.imshow(
        observations.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=num_shades, origin='lower', cmap='coolwarm',
    )
    cax = plt.axes([0.97, 2*gap+height, 0.03, height])
    cbar = plt.colorbar(h, cax=cax, label='Color cue')
    cbar.set_ticks([0, num_shades])
    idxs = (actions==1)&(rewards>0)
    h_true = ax.scatter(np.arange(num_steps)[idxs], np.zeros(sum(idxs)), color='magenta', marker='o', s=50)
    idxs = (actions==1)&(rewards<0)
    h_false = ax.scatter(np.arange(num_steps)[idxs], np.zeros(sum(idxs)), color='salmon', marker='x', s=50)
    ax.legend([h_true, h_false], ['Hit', 'Miss'], loc='upper left', fontsize=12)
    ax.spines['left'].set_visible(False)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_xticklabels([])
    ax.set_yticks([])

    ax = plt.axes([0.05, gap, 0.9, height])
    h = ax.imshow(
        probs.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
    )
    cax = plt.axes([0.97, gap, 0.03, height])
    cbar = plt.colorbar(h, cax=cax, label='Belief')
    cbar.set_ticks([0, 1])
    ax.spines['left'].set_visible(False)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    return fig

def plot_multi_box_episode(episode, num_shades=None, figsize=(5, 4)):
    num_boxes = episode['states'].shape[1]-1
    num_steps = episode['num_steps']
    states = episode['states']
    observations = episode['observations']
    actions = episode['actions']
    rewards = episode['rewards']
    has_foods = np.array(list(product(range(2), repeat=num_boxes)))
    assert not np.any(episode['q_states'][..., :2]-has_foods)
    q_probs = episode['q_probs']
    if num_shades is None:
        num_shades = observations.max()

    fig = plt.figure(figsize=figsize)
    fig_w, fig_h = figsize
    height = 0.26
    gap = (1-3*height)/4

    aspect = num_steps/num_boxes*fig_h/fig_w*(height/0.9)
    ax = plt.axes([0.05, 3*gap+2*height, 0.9, height])
    h = ax.imshow(
        states[:, :num_boxes].T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
    )
    cax = plt.axes([0.97, 3*gap+2*height, 0.03, height])
    cbar = plt.colorbar(h, cax=cax, label='Has food')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['F', 'T'])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_xticklabels([])
    ax.set_yticks(range(num_boxes))
    ax.set_yticklabels([f'Box {i}' for i in range(num_boxes)])

    aspect = num_steps/(num_boxes+1)*fig_h/fig_w*(height/0.9)
    ax = plt.axes([0.05, 2*gap+height, 0.9, height])
    # observation
    h = ax.imshow(
        observations[:, :num_boxes].T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=num_shades, origin='lower', cmap='coolwarm',
    )
    cax = plt.axes([0.97, 2*gap+height, 0.03, height])
    cbar = plt.colorbar(h, cax=cax, label='Color cue')
    cbar.set_ticks([0, num_shades])
    # action
    t = np.arange(num_steps)
    idxs, = np.nonzero((actions!=num_boxes+1)|(observations[:-1, -1]==num_boxes)) # move
    ax.scatter(t[idxs], observations[idxs+1, -1], color='blue', marker='.', s=10)
    idxs, = np.nonzero((actions==num_boxes+1)&(observations[:-1, -1]<num_boxes)&(rewards>0))
    h_true = ax.scatter(t[idxs], observations[idxs+1, -1], color='magenta', marker='o', s=50)
    idxs, = np.nonzero((actions==num_boxes+1)&(observations[:-1, -1]<num_boxes)&(rewards<0))
    h_false = ax.scatter(t[idxs], observations[idxs+1, -1], color='salmon', marker='x', s=50)
    ax.legend([h_true, h_false], ['Hit', 'Miss'], loc='upper left', fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_ylim([-0.5, num_boxes+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_xticklabels([])
    ax.set_yticks(range(num_boxes+1))
    ax.set_yticklabels([f'Box {i}' for i in range(num_boxes)]+['Center'])

    probs = np.zeros((num_steps+1, num_boxes))
    for b_idx in range(num_boxes):
        idxs = has_foods[:, b_idx]==1
        probs[:, b_idx] = q_probs[:, idxs].sum(axis=1)
    aspect = num_steps/num_boxes*fig_h/fig_w*(height/0.9)
    ax = plt.axes([0.05, gap, 0.9, height])
    h = ax.imshow(
        probs.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
    )
    cax = plt.axes([0.97, gap, 0.03, height])
    cbar = plt.colorbar(h, cax=cax, label='Belief')
    cbar.set_ticks([0, 1])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks(range(num_boxes))
    ax.set_yticklabels([f'Box {i}' for i in range(num_boxes)])
    ax.set_xlabel('Time')
    return fig
