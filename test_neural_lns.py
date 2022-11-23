import os
import tensorflow as tf
import unittest

from solving_utils import Solver


class TestSolver(unittest.TestCase):

    def test_solver_feature_extraction(self):
        root_pth = '/Users/sean/Documents/school/phd/courses/deep_learning/neural_lns'
        instance_pth = os.path.join(root_pth, 'data/instances/facilities/test_100_100_5/instance_1.lp')
        sample_pth = os.path.join(root_pth, 'data/samples/facilities/test_100_100_5/instance_1.tfrecord')
        solver = Solver()
        solver.load_model(instance_pth)
        features = solver.extract_labeled_features()
        example = solver.encode_features(features)
        with tf.io.TFRecordWriter(sample_pth) as writer:
            writer.write(example.SerializeToString())
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
