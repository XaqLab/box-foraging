import matplotlib.pyplot as plt
import numpy as np


def get_spec(name, **kwargs):
    r"""Returns environment specifications.

    The default specifications are updated by key word arguments.

    """
    if name=='boxes':
        spec = {
            'num_boxes': 2, 'num_grades': 5,
            'p_appear': 0.2, 'p_vanish': 0.05,
            'p_true': 0.8, 'p_false': 0.2,
        }
    if name=='reward':
        spec = {
            'food': 10., 'move': -2., 'fetch': -1., 'time': -0.5,
        }

    for key in spec:
        if key in kwargs:
            spec[key] = kwargs[key]
    return spec

def plot_experience(trial, figsize=(8, 2), num_grades=None):
    r"""Plots agent experience in one trial."""
    actions = trial['actions']
    rewards = trial['rewards']
    color_cues = trial['color_cues']
    agent_poss = trial['agent_poss']

    num_steps, num_boxes = color_cues.shape
    fig_w, fig_h = figsize
    aspect = num_steps/(num_boxes+1)*fig_h/fig_w*1.5
    t = np.arange(1, num_steps+1)

    fig, ax = plt.subplots(figsize=figsize)
    if num_grades is None:
        num_grades = color_cues.max()
    h = ax.imshow(
        color_cues.T, aspect=aspect, extent=[0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=num_grades, origin='lower', cmap='coolwarm',
        )
    ax.set_ylim([-0.5, num_boxes+0.5])
    idxs = (actions!=num_boxes+1)|(agent_poss==num_boxes)
    ax.scatter(t[idxs], agent_poss[idxs], color='blue', marker='.', s=10)
    idxs = (actions==num_boxes+1)&(agent_poss<num_boxes)&(rewards>0)
    h_true = ax.scatter(t[idxs], agent_poss[idxs], color='magenta', marker='o', s=50)
    idxs = (actions==num_boxes+1)&(agent_poss<num_boxes)&(rewards<0)
    h_false = ax.scatter(t[idxs], agent_poss[idxs], color='salmon', marker='x', s=50)
    plt.colorbar(h, ticks=[0, num_grades], label='color cue')
    ax.legend([h_true, h_false], ['food', 'no food'], loc='upper left')
    ax.set_xlabel('time')
    ax.set_yticks(range(num_boxes+1))
    ax.set_yticklabels([f'box {i}' for i in range(num_boxes)]+['center'])
    return fig, ax

def plot_box_states(trial, figsize=(8, 1.5)):
    r"""Plots true box states in one trial."""
    has_foods = trial['has_foods']

    num_steps, num_boxes = has_foods.shape
    fig_w, fig_h = figsize
    aspect = num_steps/num_boxes*fig_h/fig_w*1.5

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        has_foods.T, aspect=aspect, extent=[0.5, num_steps+0.5, -0.5, num_boxes-0.5],
        origin='lower', cmap='coolwarm',
        )
    ax.set_xlabel('time')
    ax.set_yticks(range(num_boxes))
    ax.set_yticklabels([f'box {i}' for i in range(num_boxes)])
    cbar = plt.colorbar(h, label='has food')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['false', 'true'])
    return fig, ax

def plot_box_beliefs(trial, figsize=(8, 1.5)):
    r"""Plots true box states in one trial."""
    box_beliefs = trial['box_beliefs']

    num_steps, num_boxes = box_beliefs.shape
    fig_w, fig_h = figsize
    aspect = num_steps/num_boxes*fig_h/fig_w*1.5

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        box_beliefs.T, aspect=aspect, extent=[-0.5, num_steps-0.5, -0.5, num_boxes-0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
        )
    ax.set_xlabel('time')
    ax.set_yticks(range(num_boxes))
    ax.set_yticklabels([f'box {i}' for i in range(num_boxes)])
    cbar = plt.colorbar(h, label='belief')
    cbar.set_ticks([0, 1])
    return fig, ax
