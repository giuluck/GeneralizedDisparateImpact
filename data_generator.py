from random import sample

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


class DataGenerator():

    def __init__(self, parents, generative_fn, prior_distr, dim=(500, 10), seed=42):
        """
        We generate synthetic data from a pre-defined graphical model.

        Parameters:
            - parents [dict]: dictionary containing the parents of each node in the graph (<node>:<list of parents>)
            - generative_fn [dict]: dictionary containing the generative function of each child node in the graph (<node>:<function>)
            - prior_distr [dict]: dictionary containing the prior distribution of each root node in the graph (<node>:{type:<name>, 'args':<distr params>})
            - dim [tuple]: default dataset dimension
            - seed [int]: seed for reproducibility
        """

        self.parents = parents
        self.generative_fn = generative_fn
        self.prior_distr = prior_distr
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)
        # Build list of nodes
        self.nodes = list(self.parents.keys())
        # Build list of edges
        self.edges = []
        for node in self.parents:
            for parent in self.parents[node]:
                self.edges.append((parent, node))
        # Nodes with at least 1 parent
        self.child_nodes = [node for node, p in self.parents.items() if len(p) > 0]
        # Nodes with no parents
        self.root_nodes = [node for node in self.nodes if node not in self.child_nodes]

    def generate(self):
        """
        Generates a new dataset with probability distribution faithful to the current graphical model.
        """

        assert len(self.generative_fn) == len(
            self.child_nodes), f"{len(self.child_nodes)} child nodes in the graph, but only {len(self.generative_fn)} generative functions specified!"
        assert len(self.prior_distr) == len(
            self.root_nodes), f"{len(self.root_nodes)} root nodes in the graph, but only {len(self.prior_distr)} prior distributions specified!"

        df_columns = []
        data = {}

        # Sample from prior distribution
        for node in self.root_nodes:
            prior = self.prior_distr[node]['name']
            args = self.prior_distr[node]['args']
            if prior == 'uniform':
                interval = args
                data[node] = self.rng.uniform(interval[0], interval[1], self.dim[0])
            elif prior == 'gauss':
                mu, sigma = args
                data[node] = self.rng.normal(mu, sigma, self.dim[0])
            elif prior == 'exp':
                beta = args[0]
                assert beta > 0, "Beta parameter must be non-negative."
                data[node] = self.rng.exponential(beta, self.dim[0])

            df_columns.append(node)

        # Populate children nodes
        while len(df_columns) < len(self.nodes):
            remaining_nodes = [node for node in self.child_nodes if node not in df_columns]
            node = sample(remaining_nodes, 1)[0]
            parents = self.parents[node]
            data[node] = self.generative_fn[node](data, parents)
            df_columns.append(node)

        data = pd.DataFrame(data, columns=df_columns)
        return data

    def view_graph(self):
        """Draw the generator graph"""
        # Build the graph
        G = nx.DiGraph(directed=True)
        # Populate edges
        G.add_edges_from(self.edges)
        # Define nodes position
        pos = nx.spring_layout(G)
        # Draw edges and nodes
        nx.draw_networkx_nodes(G, pos, edgecolors=['white'], node_color='white', node_size=100)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=12, width=1)
        nx.draw_networkx_labels(G, pos)
        plt.show()
