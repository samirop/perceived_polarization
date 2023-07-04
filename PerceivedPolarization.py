# Import Necessary Libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.algorithms import community
from scipy.stats import skewnorm

# Build social network
class SocialNetwork:
    def __init__(self, population, topology='random', network_type='random', seed=None, opinion_distribution=None):
        # Create Opinion Distribution
        # Generate the random distribution generator
        rng = np.random.default_rng(seed=seed)

        def bound_value(x):
            if x>1:
                return 1
            elif x<-1:
                return -1
            else:
                return x

        if opinion_distribution == 'uniform':
            personal_opinion = dict(enumerate(np.sort(rng.uniform(-1,1,size=population))))
        elif opinion_distribution == 'normal':
            personal_opinion = dict(enumerate(np.sort(np.vectorize(bound_value)(rng.normal(0,0.3,size=population)))))
        # Not ready yet:
        elif opinion_distribution == 'skewed':
            personal_opinion = dict(enumerate(np.sort(np.vectorize(bound_value)(skewnorm.rvs(1, loc=-0.4, scale=0.5, size=population)))))
        elif opinion_distribution == 'bimodal':
            personal_opinion = dict(enumerate(np.sort(rng.uniform(-1,1,size=population))))

        else:
            print('let s see')

        def make_o(size, factor):
            o = np.random.normal(size = size, loc = factor, scale = 0.2) # Make a random normal distribution with std = 0.2
            o[o > 1] = 1 - (o[o > 1] - 1)                                # Mirror values out of the +/-1 bounds to be inbound
            o = np.concatenate([o, -o])                                  # Create the negative side of the distribution
            o.sort()                                                     # Sort, this will create community homophily in the SBM
            return {i: o[i] for i in range(o.shape[0])}                  # Transform into a dictionary, which si the input needed by the function

        # Create the network


        # Add properties to the network ?


        if network_type == 'random':
            self.G = nx.generators.random_graphs.erdos_renyi_graph(population, 0.5)
        elif network_type == 'scale-free':
            self.G = nx.generators.random_graphs.scale_free_graph(population)
        elif network_type == 'small-world':
            self.G = nx.generators.random_graphs.watts_strogatz_graph(population, 4, 0.3)
        elif network_type == 'community':
            self.G = nx.generators.random_graphs.erdos_renyi_graph(population, 0.5)

            # Detect communities using the Girvan-Newman algorithm
            communities = community.girvan_newman(self.G)

            # Select the top-level communities
            top_level_communities = next(communities)

            # Assign community labels to nodes
            for i, comm in enumerate(top_level_communities):
                for node in comm:
                    self.G.nodes[node]['community'] = i
        else:
            raise ValueError("Invalid network_type specified.")

            # Initialize the personal opinions

        for node in self.G.nodes:
            self.G.nodes[node]['personal_opinion'] = np.random.uniform(-1, 1)
            self.G.nodes[node]['perceived_opinion'] = 0
            self.G.nodes[node]['loudness'] = 1 # Not included yet in the model

class PerceivedPolarization:
    def __init__(self, socialnetwork):
        self.G = socialnetwork.G

    def node_perceived_opinion(self, node):
        neighbor_opinions = [self.G.nodes[neighbor]['personal_opinion'] for neighbor in self.G.neighbors(node)]
        return sum(neighbor_opinions) / len(neighbor_opinions) if neighbor_opinions else 0

    def calculate_polarization(self):
        return np.mean([abs(self.G.nodes[node]['personal_opinion'] - self.G.nodes[node]['perceived_opinion']) for node in self.G.nodes])

    def global_perceived_opinion(self):
        for node in self.G.nodes:
            self.G.nodes[node]['perceived_opinion'] = self.node_perceived_opinion(node)

    def draw_graph(self):
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        pos = nx.spring_layout(self.G)

        personal_opinions = [self.G.nodes[node]['personal_opinion'] for node in self.G.nodes]
        personal_opinions_colors = [cm.RdYlBu(op) for op in personal_opinions]
        nx.draw(self.G, pos, node_color=personal_opinions_colors, node_size=300, alpha=0.7, linewidths=0.5, edgecolors='k', ax=axs[0])
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlBu, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axs[0])
        axs[0].set_title('Personal Opinions')

        perceived_opinions = [self.G.nodes[node]['perceived_opinion'] for node in self.G.nodes]
        perceived_opinions_colors = [cm.RdYlBu(op) for op in perceived_opinions]
        nx.draw(self.G, pos, node_color=perceived_opinions_colors, node_size=300, alpha=0.7, linewidths=0.5, edgecolors='k', ax=axs[1])
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlBu, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axs[1])
        axs[1].set_title('Perceived Opinions')

        plt.show()
