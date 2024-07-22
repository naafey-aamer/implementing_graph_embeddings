import pandas as pd
import random

class RandomWalk:
    def __init__(self, graph_df):
        self.graph_df = graph_df

    def random_walk(self, start_node, walk_length):
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
            next_node = random.choice(neighbors_list)
            walk.append(next_node)
        return walk
