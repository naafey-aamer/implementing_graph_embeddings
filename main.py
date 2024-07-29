import pandas as pd
import networkx as nx
import numpy as np
from src.utils import evaluate_embeddings
from src.random_walk import RandomWalk
from src.deep_walk import DeepWalk
from src.node2vec import Node2Vec
from src.triplet_walk import TripletWalk

def generate_large_graph(num_nodes, prob):
    G = nx.erdos_renyi_graph(num_nodes, prob)
    adj_list = {str(node): list(map(str, neighbors)) for node, neighbors in G.adjacency()}
    graph_data = {'node': list(adj_list.keys()), 'neighbors': list(adj_list.values())}
    return pd.DataFrame(graph_data)

def generate_walks(model, start_node, walk_length, num_walks):
    return [model.random_walk(start_node, walk_length) for _ in range(num_walks)]

def train_and_evaluate(model, walks, embed_size, num_epochs, learning_rate, graph_df):
    skip_gram_pairs = model.generate_skip_gram_data(walks, window_size=2)
    embeddings, node_index = model.train_skip_gram(skip_gram_pairs, embed_size, num_epochs, learning_rate)
    # print(embeddings)
    evaluate_embeddings(embeddings, node_index, graph_df)

def main():
    graph_df = generate_large_graph(num_nodes=300, prob=0.05)
    start_node = graph_df['node'].iloc[0]
    walk_length = 20
    num_walks = 20
    embed_size = 512
    num_epochs = 100
    learning_rate = 0.001
    p = 1
    q = 1

    # Random Walk
    rw = RandomWalk(graph_df)
    walks_rw = generate_walks(rw, start_node, walk_length, num_walks)
    dw = DeepWalk(graph_df)
    print("Evaluating DeepWalk embeddings:")
    train_and_evaluate(dw, walks_rw, embed_size, num_epochs, learning_rate, graph_df)

    # Node2Vec
    nv = Node2Vec(graph_df, p, q)
    walks_nv = [nv.biased_random_walk(start_node, walk_length) for _ in range(num_walks)]
    print("Evaluating Node2Vec embeddings:")
    train_and_evaluate(dw, walks_nv, embed_size, num_epochs, learning_rate, graph_df)

    # TripletWalk
    tw = TripletWalk(graph_df, p, q)
    walks_tw = [tw.triplet_walk(start_node, walk_length) for _ in range(num_walks)]
    print("Evaluating Triple Walk embeddings:")
    train_and_evaluate(dw, walks_tw, embed_size, num_epochs, learning_rate, graph_df)

if __name__ == "__main__":
    main()
