import tensorflow as tf
import unittest
from unittest.mock import patch

from data_utils import SCIP_FEATURE_EXTRACTION_PARAMS
from solving_utils import Solver


class TestSolver(unittest.TestCase):

    def test_solver_feature_extraction(self):
        solver = Solver()
        solver.load_model('/Users/sean/Documents/school/phd/courses/deep_learning/neural_lns/data/instances/facilities/test_100_100_5/instance_1.lp')
        example = solver.extract_labeled_features()
        with tf.io.TFRecordWriter('/Users/sean/Documents/school/phd/courses/deep_learning/neural_lns/data/samples/facilities/test_100_100_5/instance_1.tfrecord') as writer:
            writer.write(example.SerializeToString())
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
