# Import necessary modules 
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SocialNetwork():
    """Creates the social network and opinion distribution
    """
    def __init__(self, population, topology = 'random',opinion_distribution=None,whateveryoucomeupwith=None):
        # Create the network
        self.G = nx.generators.classic.complete_graph(population) if topology == 'complete' else nx.generators.random_graphs.erdos_renyi_graph(population, 0.5)
        
        # Initialize the personal opinions
        for node in self.G.nodes:
            self.G.nodes[node]['personal_opinion'] = np.random.uniform(-1, 1)
            self.G.nodes[node]['perceived_opinion'] = 0
            self.G.nodes[node]['loudness'] = 1 # Not included yet in the model 


class PerceivedPolarization():
    def __init__(self, socialnetwork):
        # Create the network
        self.G = socialnetwork.G

    def node_perceived_opinion(self, node):
        # replace with your preferred method of calculating perceived opinion
        neighbor_opinions = [self.G.nodes[neighbor]['personal_opinion'] for neighbor in self.G.neighbors(node)]
        return sum(neighbor_opinions) / len(neighbor_opinions) if neighbor_opinions else 0

    def global_perceived_opinion(self):
        # Calculate the perceived opinion for every onde
        for node in self.G.nodes:
            self.G.nodes[node]['perceived_opinion'] = self.node_perceived_opinion(node)
            
    def calculate_polarization(self):
        # Here add Marilena's models or Standard Deviation =) 
        return np.mean([abs(self.G.nodes[node]['personal_opinion'] - self.G.nodes[node]['perceived_opinion']) for node in self.G.nodes])

    def draw_graph(self):
        # Still need to fix colors =) 
        fig, axs = plt.subplots(2, 1, figsize=(8,12))

        # Layout for the nodes
        pos = nx.spring_layout(self.G)

        # Personal opinions
        personal_opinions = [self.G.nodes[node]['personal_opinion'] for node in self.G.nodes]
        personal_opinions_colors = [cm.RdYlBu(op) for op in personal_opinions]
        nx.draw(self.G, pos, node_color=personal_opinions_colors, node_size=300, alpha=0.7, linewidths=0.5, edgecolors='k', ax=axs[0])
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlBu, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axs[0])
        axs[0].set_title('Personal Opinions')

        # Perceived opinions 
        perceived_opinions = [self.G.nodes[node]['perceived_opinion'] for node in self.G.nodes]
        perceived_opinions_colors = [cm.RdYlBu(op) for op in perceived_opinions]
        nx.draw(self.G, pos, node_color=perceived_opinions_colors, node_size=300, alpha=0.7, linewidths=0.5, edgecolors='k', ax=axs[1])
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlBu, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axs[1])
        axs[1].set_title('Perceived Opinions')

        plt.show()
        
    def analytics(self):
        # Add different analytics =) 
        pass 