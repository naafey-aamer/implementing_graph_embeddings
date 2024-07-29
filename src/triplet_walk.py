from .node2vec import Node2Vec
import numpy as np

class TripletWalk(Node2Vec):
    def __init__(self, graph_df, p, q):
        super().__init__(graph_df, p, q)

    def triplet_walk(self, start_node, walk_length):
        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = self.graph_df.loc[self.graph_df['node'] == current_node, 'neighbors']
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
                elif neighbor in self.graph_df.loc[self.graph_df['node'] == walk[-2], 'neighbors'].values[0] if len(walk) > 1 else []:
                    probabilities.append(1)
                else:
                    probabilities.append(1 / self.q)
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            next_node = np.random.choice(neighbors_list, p=probabilities)
            walk.append(next_node)
        return walk
