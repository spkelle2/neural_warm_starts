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
"""A light GNN model for training NeuralLNS."""

from typing import List

from graph_nets import graphs
import sonnet as snt
import tensorflow.compat.v2 as tf

import layer_norm

GT_SPEC = graphs.GraphsTuple(
    nodes=tf.TensorSpec(shape=(None, 34), dtype=tf.float32, name='nodes'),
    edges=tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='edges'),
    receivers=tf.TensorSpec(shape=(None,), dtype=tf.int64, name='receivers'),
    senders=tf.TensorSpec(shape=(None,), dtype=tf.int64, name='senders'),
    globals=tf.TensorSpec(shape=(), dtype=tf.float32, name='globals'),
    n_node=tf.TensorSpec(shape=(None,), dtype=tf.int32, name='n_node'),
    n_edge=tf.TensorSpec(shape=(None,), dtype=tf.int32, name='n_edge'))


def get_adjacency_matrix(graph: graphs.GraphsTuple) -> tf.SparseTensor:
    upper = tf.stack([graph.senders, graph.receivers], axis=1)
    lower = tf.stack([graph.receivers, graph.senders], axis=1)
    indices = tf.concat([upper, lower], axis=0)
    values = tf.squeeze(tf.concat([graph.edges, graph.edges], axis=0))
    dense_shape = tf.cast(
        tf.stack([graph.n_node[0], graph.n_node[0]], axis=0),
        dtype=tf.int64)
    adj = tf.sparse.SparseTensor(indices, values, dense_shape)
    return tf.sparse.reorder(adj)


class LightGNNLayer(snt.Module):
    """A single layer of a GCN."""

    def __init__(self,
                 node_model_hidden_sizes: List[int],
                 name=None):
        super(LightGNNLayer, self).__init__(name=name)
        self._node_model_hidden_sizes = node_model_hidden_sizes

    @snt.once
    def _initialize(self):
        """ a GCN layer is a single layer MLP (i.e. input and output nodes) with connections matching
        the adjacency matrix. First layer inputs represent features of each node in a graph"""
        self._mlp = snt.nets.MLP(self._node_model_hidden_sizes,
                                 activate_final=False)

    def __call__(self,
                 input_nodes: tf.Tensor,
                 adj_mat: tf.SparseTensor,
                 is_training: bool) -> tf.Tensor:
        self._initialize()
        updated_nodes = self._mlp(input_nodes)  # pass through fully connected MLP layer
        # keep connections corresponding to adjacency matrix
        combined_nodes = tf.sparse.sparse_dense_matmul(adj_mat, updated_nodes)
        return combined_nodes


class LightGNN(snt.Module):
    """A stack of LightGNNLayers."""

    def __init__(self,
                 n_layers: int,
                 node_model_hidden_sizes: List[int],
                 output_model_hidden_sizes: List[int],
                 dropout: float = 0.0,
                 name=None,
                 **unused_args):
        super(LightGNN, self).__init__(name=name)

        self._n_layers = n_layers
        self._node_model_hidden_sizes = node_model_hidden_sizes
        self._output_model_hidden_sizes = output_model_hidden_sizes
        self._dropout = dropout

    @snt.once
    def _initialize(self):
        # builds the graph convolutional network (eqns 2-4 in paper) to compute H dim vector for each node in milp graph
        self._layers = []
        for i in range(self._n_layers):
            layer = LightGNNLayer(
                self._node_model_hidden_sizes,
                name='layer_%d' % i)
            # Wrapper to apply layer normalisation, residual (skip) connection, and dropout
            layer = layer_norm.ResidualDropoutWrapper(
                layer, dropout_rate=self._dropout)
            self._layers.append(layer)

        # linear model before GCN (todo: why?)
        self._input_embedding_model = snt.Linear(
            self._node_model_hidden_sizes[-1], name='input_embedding')
        # the mlp that the GCN then feeds
        self.output_model = snt.nets.MLP(self._output_model_hidden_sizes,
                                         name='output_model')

    def encode_graph(self,
                     graph: graphs.GraphsTuple,
                     is_training: bool) -> tf.Tensor:
        """This converts an input graph to "node embeddings" i.e. pass through GCN but not output MLP"""
        self._initialize()
        adj = get_adjacency_matrix(graph)  # creates adjacency matrix A
        nodes = self._input_embedding_model(graph.nodes)
        for layer in self._layers:
            nodes = layer(nodes, adj, is_training=is_training)

        return nodes  # this is the "node embedding", the output of the GCN

    def __call__(self,
                 graph: graphs.GraphsTuple,
                 is_training: bool,
                 node_indices: tf.Tensor,
                 labels: tf.Tensor,
                 **unused_args) -> tf.Tensor:
        # label data dimensions (todo: number of variables and bits per variable?)
        n = tf.shape(labels)[0]
        b = tf.shape(labels)[1]
        # take MILP graph and create an H dimensional vector for each node
        # each vector is a node embedding and is the result of passing through a GCN
        # is_training option applies drop out
        nodes = self.encode_graph(graph, is_training)  # pass through GCN. is_training applies dropout
        # sonnet infers input sizes which is how we go from 64 to 32 to 1 length vectors for each node via mlp below
        all_logits = self.output_model(nodes)
        # subselect bit predictions for nodes corresponding to variables (todo: how do we make sure to get all predictions when multiple bits/variable?)
        logits = tf.expand_dims(tf.gather(all_logits, node_indices), axis=-1)
        logits = tf.broadcast_to(logits, [n, b, 1])

        return logits  # \mu_d is the sigmoid of logit[d], i.e. p(x_d=1 | MILP)

    @tf.function(input_signature=[
        GT_SPEC,
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='node_indices')
    ])
    def greedy_sample(self, graph, node_indices):
        nodes = self.encode_graph(graph, False)
        logits = self.output_model(nodes)
        probas = tf.math.sigmoid(tf.gather(logits, node_indices))
        sample = tf.round(probas)
        return sample, probas

    @tf.function(input_signature=[
        GT_SPEC,
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32)
    ])
    def predict_logits(self,
                       graph: graphs.GraphsTuple,
                       node_indices: tf.Tensor,
                       labels: tf.Tensor) -> tf.Tensor:
        return self(graph, False, node_indices, labels)

    def save_model(self, output_dir: str):
        """Saves a model to output directory."""
        tf.saved_model.save(
            self, output_dir, signatures={'greedy_sample': self.greedy_sample})


class NeuralLnsLightGNN(LightGNN):
    """A stack of LightGNNLayers."""

    def __init__(self,
                 n_layers: int,
                 node_model_hidden_sizes: List[int],
                 output_model_hidden_sizes: List[int],
                 dropout: float = 0.0,
                 name=None,
                 **unused_args):
        super().__init__(n_layers, node_model_hidden_sizes,
                         output_model_hidden_sizes,
                         dropout)

    @tf.function(input_signature=[
        GT_SPEC,
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='node_indices')
    ])
    def greedy_sample(self, graph, node_indices):
        nodes = self.encode_graph(graph, False)
        logits = self.output_model(nodes)
        probas = tf.math.sigmoid(tf.gather(logits, node_indices))  # probability each bit is one
        sample = tf.round(probas)
        return sample, probas  # sample is boolean value with higher probability

    @tf.function(input_signature=[
        GT_SPEC,
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32)
    ])
    def predict_logits(self,
                       graph: graphs.GraphsTuple,
                       node_indices: tf.Tensor,
                       labels: tf.Tensor) -> tf.Tensor:
        return self(graph, False, node_indices, labels)


def get_model(**params):
    return NeuralLnsLightGNN(**params)
