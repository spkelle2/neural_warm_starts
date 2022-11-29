import os
import tensorflow as tf

from solvers import Solver


def main(instance_type: str):
    """ Take a collection of directories and return

    :param instance_type:
    :return:
    """

    # data checks
    assert instance_type in ['facilities', 'schedules', 'cauctions']

    # declare script level variables
    instance_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'data/instances', instance_type)
    sample_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'data/samples', instance_type)

    for subdir_name in os.listdir(instance_root):
        if 'test' in subdir_name:
            continue
        for i, file_name in enumerate(os.listdir(os.path.join(instance_root, subdir_name))):

            print(f'generating directory {subdir_name} instance {i + 1}')

            # declare loop level variables
            file_root = file_name.split('.')[0]
            instance_pth = os.path.join(instance_root, subdir_name, file_name)
            sample_pth = os.path.join(sample_root, subdir_name, f'{file_root}.tfrecord')
            solver = Solver()

            # create features for this instance
            solver.load_model(instance_pth)
            features = solver.extract_labeled_features()
            example = solver.encode_features(features)
            with tf.io.TFRecordWriter(sample_pth) as writer:
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main(instance_type='cauctions')
