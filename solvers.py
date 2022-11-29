# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common utilities for Solver."""

import abc
import enum
import gurobipy as gu
from math import isclose
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from typing import Any, Dict, Optional

import data_utils
import sampling


class SolverState(enum.Enum):
    INIT = 0
    MODEL_LOADED = 1
    FINISHED = 2


type_map = {gu.GRB.BINARY: 0, gu.GRB.INTEGER: 1, gu.GRB.CONTINUOUS: 3}
status_map = {gu.GRB.NONBASIC_LOWER: 0, gu.GRB.BASIC: 1, gu.GRB.NONBASIC_UPPER: 2, gu.GRB.SUPERBASIC: 3}


def _bytes_single_value_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


feature_type = {
    'constraint_features': _bytes_single_value_feature,
    'edge_features': _bytes_single_value_feature,
    'edge_indices': _bytes_single_value_feature,
    'variable_features': _bytes_single_value_feature,
    'variable_lbs': _float_feature,
    'variable_ubs': _float_feature,
    'constraint_feature_names': _bytes_single_value_feature,
    'variable_feature_names': _bytes_single_value_feature,
    'edge_features_names': _bytes_single_value_feature,
    'variable_names': _bytes_feature,
    'binary_variable_indices': _int64_feature,
    'all_integer_variable_indices': _int64_feature,
    'model_maximize': _int64_feature,
    'best_solution_labels': _float_feature,
}


class Solver(abc.ABC):
    """Wrapper around a given classical MIP solver.

    This class contains the API needed to communicate with a MIP solver, e.g.
    Gurobi.
    """

    def __init__(self, sampler: sampling.RepeatedCompetitionSampler = None):
        super().__init__()
        self.m = None
        self.sampler = sampler

    def load_model(self, mip_pth: str, silence: bool = True) -> SolverState:
        """Loads a MIP model into the solver."""
        env = gu.Env(empty=True)
        if silence:
            env.setParam('OutputFlag', 0)
        env.start()
        self.m = gu.read(mip_pth, env=env)
        self.m.setParam('MIPGap', 0.01)
        return SolverState.MODEL_LOADED

    def solve(self, **kwargs) -> None:
        """Solves the loaded MIP model."""
        self.m.optimize()
        assert self.m.status == gu.GRB.OPTIMAL

    def get_best_solution(self) -> Optional[Any]:
        """Returns the best solution found from the last solve call."""
        assert self.m.status == gu.GRB.OPTIMAL
        assert [v.index for v in self.m.getVars()] == list(range(self.m.NumVars)), \
            "we need to assume that variables are in the same order as columns"

        return np.array([v.x for v in self.m.getVars()]).reshape(-1, 1)

    def extract_lp_features_at_root(self, training: bool = True) -> Dict[str, Any]:
        """Returns a dictionary of root node features."""

        assert [v.index for v in self.m.getVars()] == list(range(self.m.NumVars)), \
            "we need to assume that variables are in the same order as columns"

        assert [c.index for c in self.m.getConstrs()] == list(range(len(self.m.getConstrs()))), \
            "we need to assume that constraints are in the same order as rows"

        # some values we'll reuse
        objective = np.array([v.Obj for v in self.m.getVars()])
        obj_norm = np.linalg.norm(objective)
        obj_norm = 1 if obj_norm <= 0 else obj_norm
        A = self.m.getA().toarray()
        row_norms = np.linalg.norm(A, axis=1)
        row_norms[row_norms == 0] = 1

        # features already available
        lbs = np.array([max(v.LB, -1e10) for v in self.m.getVars()])
        ubs = np.array([min(v.UB, 1e10) for v in self.m.getVars()])
        binary_idxs = np.array([v.index for v in self.m.getVars() if v.VType == gu.GRB.BINARY],
                               dtype=np.int64)
        integer_idxs = np.array([v.index for v in self.m.getVars() if v.VType == gu.GRB.INTEGER],
                                dtype=np.int64)
        variable_names = [v.VarName for v in self.m.getVars()]
        model_maximize = np.array([int(self.m.ModelSense == gu.GRB.MAXIMIZE)], dtype=np.int64)

        # solve the root relaxation
        relax = self.m.relax()
        relax.optimize()

        # Column features - for unsures I think 0 is reasonable since this is the first LP solved
        col_feats = {}
        col_feats['age'] = np.zeros((self.m.NumVars, 1))  # unsure how this is calculated
        col_feats['avg_inc_val'] = np.zeros((self.m.NumVars, 1))  # unsure how this is calculated
        col_feats['basis_status'] = np.zeros((self.m.NumVars, 4))  # LOWER BASIC UPPER ZERO
        status_codes = [status_map[v.VBasis] for v in relax.getVars()]
        col_feats['basis_status'][np.arange(self.m.NumVars), status_codes] = 1
        col_feats['coef_normalized'] = objective.reshape(-1, 1) / obj_norm
        col_feats['has_lb'] = (lbs > -float('inf')).astype(np.int64).reshape(-1, 1)
        col_feats['has_ub'] = (ubs < float('inf')).astype(np.int64).reshape(-1, 1)
        col_feats['inc_val'] = np.zeros((self.m.NumVars, 1))  # unsure how this is calculated
        reduced_costs = [v.RC for v in relax.getVars()]
        col_feats['reduced_cost'] = np.array(reduced_costs).reshape(-1, 1) / obj_norm
        col_feats['sol_frac'] = np.array([v.x % 1 for v in relax.getVars()]).reshape(-1, 1)
        # continuous have no fractionality
        col_feats['sol_frac'][[v.VType == gu.GRB.CONTINUOUS for v in self.m.getVars()]] = 0
        at_lb = [int(isclose(v.LB, v.x, abs_tol=1e-4)) for v in relax.getVars()]
        col_feats['sol_is_at_lb'] = np.array(at_lb).reshape(-1, 1)
        at_ub = [int(isclose(v.UB, v.x, abs_tol=1e-4)) for v in relax.getVars()]
        col_feats['sol_is_at_ub'] = np.array(at_ub).reshape(-1, 1)
        col_feats['sol_val'] = np.array([v.x for v in relax.getVars()]).reshape(-1, 1)
        # changed columns from 4 to 11 to account for 7 extra columns needed for LNS incumbents
        # leave the 7 extra columns as zero since we don't use LNS
        col_feats['type'] = np.zeros((self.m.NumVars, 11))  # BINARY INTEGER IMPLINT CONTINUOUS
        type_codes = [type_map[v.VType] for v in self.m.getVars()]
        col_feats['type'][np.arange(self.m.NumVars), type_codes] = 1

        col_feat_names = [
            [k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])]
            for k, v in col_feats.items()
        ]
        col_feat_names = [n for names in col_feat_names for n in names]
        col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)
        if training:
            col_feat_vals = col_feat_vals.astype(np.float32)

        # Row features
        row_feats = {}
        # all have same age, which is 0th LP relaxation
        row_feats['age'] = np.zeros(self.m.NumConstrs).reshape(-1, 1)
        rhs = np.array([c.RHS for c in self.m.getConstrs()])
        row_feats['bias'] = np.array(rhs / row_norms).reshape(-1, 1)
        duals = np.array([c.Pi for c in relax.getConstrs()])
        row_feats['dualsol_val_normalized'] = (duals / (row_norms * obj_norm)).reshape(-1, 1)
        row_feats['is_tight'] = np.array([int(isclose(c.Slack, 0, abs_tol=1e-4))
                                          for c in relax.getConstrs()], dtype=np.int64).reshape(-1, 1)
        row_feats['obj_cosine_similarity'] = (np.dot(A, objective) / (row_norms * obj_norm)).reshape(-1, 1)

        row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                          row_feats.items()]
        row_feat_names = [n for names in row_feat_names for n in names]
        row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)
        if training:
            row_feat_vals = row_feat_vals.astype(np.float32)

        # Edge features - normalize the coef matrix
        coef_matrix = sp.coo_matrix(self.m.getA() / row_norms.reshape(-1, 1))
        edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
        edge_feats = {'coef_normalized': coef_matrix.data.reshape(-1, 1)}

        edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                           edge_feats.items()]
        edge_feat_names = [n for names in edge_feat_names for n in names]
        edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs]).T.astype(np.int64)
        edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)
        if training:
            edge_feat_vals = edge_feat_vals.astype(np.float32)

        features = {
            'edge_features_names': edge_feat_names,
            'best_solution_labels': np.array([]),
            'edge_features': edge_feat_vals,
            'binary_variable_indices': binary_idxs,
            'variable_ubs': ubs,
            'edge_indices': edge_feat_indices,
            'variable_feature_names': col_feat_names,
            'model_maximize': model_maximize,
            'constraint_features': row_feat_vals,
            'variable_lbs': lbs,
            'all_integer_variable_indices': integer_idxs,
            'variable_names': variable_names,
            'variable_features': col_feat_vals,
            'constraint_feature_names': row_feat_names
        }

        return features

    def encode_features(self, features):

        # encode string and 2-D arrays to bytes
        features['variable_names'] = [n.encode('utf-8') for n in features['variable_names']]
        for name in ['edge_features_names', 'variable_feature_names', 'constraint_feature_names']:
            features[name] = ','.join([n for n in features[name]]).encode('utf-8')
        for name in ['edge_features', 'variable_features', 'constraint_features', 'edge_indices']:
            features[name] = tf.io.serialize_tensor(features[name])

        # convert all values to tf.train.Feature
        features = {k: feature_type[k](v) for k, v in features.items()}
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example

    def extract_labeled_features(self):
        """Create the root feature set along with optimal solution"""
        features = self.extract_lp_features_at_root()
        self.solve()
        features['best_solution_labels'] = self.get_best_solution()
        return features


class DirectSolver(Solver):
    """Agent that solves MIP directly with Gurobi."""

    def solve(self) -> Dict[str, Any]:
        self.m.optimize()
        stats = {
            'direct optimal': self.m.status == gu.GRB.OPTIMAL,
            'direct time': self.m.runtime
        }
        return stats


class NeuralDivingSolver(Solver):
    """Solver that implements Neural Diving with Gurobi."""

    def solve(self, num_unassigned_vars: int) -> Dict[str, Any]:
        # using partial mip starts is really slow so trying to provide a full start instead
        # capture which variables to fix
        features = self.extract_lp_features_at_root(training=False)
        graphs_tuple = data_utils.get_graphs_tuple(features)
        assignment = self.sampler.sample(graphs_tuple, features['variable_names'],
                                         features['variable_lbs'], features['binary_variable_indices'],
                                         num_unassigned_vars=num_unassigned_vars)

        # create and solve the submip
        sub_m = self.m.copy()  # will know if copied parameters if silences
        for name, val in zip(assignment.names, assignment.upper_bounds):
            sub_m.getVarByName(name).ub = val
        sub_m.optimize()

        # provide its solution to the original mip if feasible
        if sub_m.status == gu.GRB.OPTIMAL:
            self.m.NumStart = 1
            self.m.params.StartNumber = 0
            for v in self.m.getVars():
                v.start = sub_m.getVarByName(v.varName).X
        else:
            print('prediction failed to find feasible solution')

        self.m.optimize()
        stats = {
            'neural diving optimal': self.m.status == gu.GRB.OPTIMAL,
            'neural diving time': self.m.runtime + sub_m.runtime,
            'neural diving objective': self.m.objVal,
            'submip optimal': sub_m.status == gu.GRB.OPTIMAL,
            'submip time': sub_m.runtime,
            'submip objective': sub_m.objVal if sub_m.status == gu.GRB.OPTIMAL else None,
        }
        return stats
