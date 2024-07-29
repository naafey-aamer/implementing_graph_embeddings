import numpy as np
from .random_walk import RandomWalk

class DeepWalk(RandomWalk):
    def __init__(self, graph_df):
        super().__init__(graph_df)

    def generate_skip_gram_data(self, walks, window_size):
        pairs = []
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, window_size + 1):
                    if i - j >= 0:
                        pairs.append((walk[i], walk[i - j]))
                    if i + j < len(walk):
                        pairs.append((walk[i], walk[i + j]))
        return pairs

    def safe_sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


    def train_skip_gram(self, pairs, embed_size, num_epochs, learning_rate):
        nodes = list(set([node for pair in pairs for node in pair]))
        node_index = {node: idx for idx, node in enumerate(nodes)}

        num_nodes = len(nodes)
        W = np.random.randn(num_nodes, embed_size) * 0.01
        W_context = np.random.randn(num_nodes, embed_size) * 0.01

        for epoch in range(num_epochs):
            for target, context in pairs:
                target_idx = node_index[target]
                context_idx = node_index[context]

                # Compute score using dot product
                score = np.dot(W[target_idx], W_context[context_idx])

                # Apply the safe sigmoid function
                sigmoid_score = self.safe_sigmoid(score)
                error = 1 - sigmoid_score

                # Update weights
                W[target_idx] -= learning_rate * error * W_context[context_idx]
                W_context[context_idx] -= learning_rate * error * W[target_idx]

        return W, node_index