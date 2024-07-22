from .random_walk import RandomWalk
import numpy as np

class Node2Vec(RandomWalk):
    def __init__(self, graph_df, p, q):
        super().__init__(graph_df)
        self.p = p
        self.q = q

    def biased_random_walk(self, start_node, walk_length):
        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = self.graph_df[self.graph_df['node'] == current_node]['neighbors']
            if neighbors.empty:
                print(f"No neighbors found for node {current_node}")
                break
            neighbors_list = neighbors.values[0]
            if isinstance(neighbors_list, str):
                neighbors_list = neighbors_list.split(';')
            if not neighbors_list:
                print(f"No neighbors found for node {current_node}")
                break

            probabilities = []
            for neighbor in neighbors_list:
                if neighbor == walk[-2] if len(walk) > 1 else None:
                    probabilities.append(1 / self.p)
                elif neighbor in self.graph_df[self.graph_df['node'] == walk[-2]]['neighbors'].values[0] if len(walk) > 1 else []:
                    probabilities.append(1)
                else:
                    probabilities.append(1 / self.q)
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            next_node = np.random.choice(neighbors_list, p=probabilities)
            walk.append(next_node)
        return walk

    def tanh(self, x):
        return np.tanh(x)

    def train_skip_gram(self, pairs, embed_size, num_epochs, learning_rate):
        nodes = list(set([node for pair in pairs for node in pair]))
        node_index = {node: idx for idx, node in enumerate(nodes)}

        num_nodes = len(nodes)
        W = np.random.rand(num_nodes, embed_size)
        W_context = np.random.rand(num_nodes, embed_size)

        for epoch in range(num_epochs):
            for target, context in pairs:
                target_idx = node_index[target]
                context_idx = node_index[context]

                # Compute score using dot product
                score = np.dot(W[target_idx], W_context[context_idx])

                # Apply the tanh function
                tanh_score = self.tanh(score)
                error = 1 - tanh_score

                # Update weights
                W[target_idx] -= learning_rate * error * W_context[context_idx]
                W_context[context_idx] -= learning_rate * error * W[target_idx]

        return W, node_index
