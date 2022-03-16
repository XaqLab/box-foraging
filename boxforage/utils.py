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
            'food': 10., 'move': -2., 'time': -0.5,
        }

    for key in spec:
        if key in kwargs:
            spec[key] = kwargs[key]
    return spec
