import os
import pandas as pd
import re

from sampling import RepeatedCompetitionSampler
from solvers import NeuralDivingSolver, DirectSolver


def main(instance_type: str, unassigned_factor: int):
    """ Compares the solve time of a discrete optimization solver when warm started
    with a predicted solution vs cold started

    :param instance_type: which problem type to use
    :param unassigned_factor: multiple of number of solutions that will take value 1
    to leave unassigned
    :return:
    """

    # data checks
    assert instance_type in ['facilities', 'schedules', 'cauctions']
    assert isinstance(unassigned_factor, (int, float)) and unassigned_factor > 0

    # declare script level variables
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'models_{instance_type}')
    instance_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'data/instances', instance_type)
    eval_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'data/evaluations', f'{instance_type}.csv')
    sampler = RepeatedCompetitionSampler(model_dir)
    data = {}
    num_unassigned_vars = None

    for subdir_name in os.listdir(instance_root):
        if ('transfer' not in subdir_name) and ('train' not in subdir_name):
            continue

        # set number of unassigned variables
        if instance_type == 'facilities':
            n_cust, n_facilities, ratio = [int(n) for n in re.match(r'^\w+_(\d+)_(\d+)_(\d+)$', subdir_name).groups()]
            # give ourselves <unassigned_factor> as many warehouses as cover capacity (2 is good)
            num_unassigned_vars = int(unassigned_factor * ratio * n_cust / n_facilities)
        if instance_type == 'schedules':
            n_weeks, n_teams, ratio = [int(n) for n in re.match(r'^\w+_(\d+)_(\d+)_(\d+)$', subdir_name).groups()]
            # each week we have n_teams/2 games, so give ourselves <unassigned_factor>x as many options (2 is good)
            num_unassigned_vars = int(unassigned_factor * n_weeks * n_teams)
        if instance_type == 'cauctions':
            n_items, n_bids = [int(n) for n in re.match(r'^\w+_(\d+)_(\d+)$', subdir_name).groups()]
            # for each item we accept 1 bid, so select <unassigned_factor>x as many options (2 is good)
            num_unassigned_vars = int(unassigned_factor * n_items)

        # compare warm and cold starts
        for i, file_name in enumerate(os.listdir(os.path.join(instance_root, subdir_name))):
            if i >= 10:
                continue
            print(f'evaluating directory {subdir_name} instance {i + 1}')
            file_root = file_name.split('.')[0]
            instance_pth = os.path.join(instance_root, subdir_name, file_name)

            # solvers for each instance
            ds = DirectSolver()
            ds.load_model(instance_pth, silence=True)
            direct_data = ds.solve()

            nds = NeuralDivingSolver(sampler=sampler)
            nds.load_model(instance_pth, silence=True)
            neural_diving_data = nds.solve(num_unassigned_vars=num_unassigned_vars)
            data[subdir_name, file_root] = {**direct_data, **neural_diving_data}

    # save the runs
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['instance set', 'instance name'] + list(df.columns[2:])
    df.to_csv(eval_pth, index=False)
    print()


if __name__ == '__main__':
    main('cauctions', unassigned_factor=2)
