import pandas as pd
import networkx as nx
import numpy as np
from src.utils import evaluate_embeddings
from src.random_walk import RandomWalk
from src.deep_walk import DeepWalk
from src.node2vec import Node2Vec
from src.triplet_walk import TripletWalk

# Generate a large synthetic graph using NetworkX
def generate_large_graph(num_nodes=1000, prob=0.01):
    G = nx.erdos_renyi_graph(num_nodes, prob)
    adj_list = {str(node): list(map(str, neighbors)) for node, neighbors in G.adjacency()}
    graph_data = {'node': list(adj_list.keys()), 'neighbors': list(adj_list.values())}
    return pd.DataFrame(graph_data)

# Create a large graph DataFrame
graph_df = generate_large_graph(num_nodes=1000, prob=0.01)

# Parameters
start_node = graph_df['node'].iloc[0]
walk_length = 20
num_walks = 20
embed_size = 128
num_epochs = 500
learning_rate = 0.05
p = 1
q = 1

# Random Walk
rw = RandomWalk(graph_df)
walks = [rw.random_walk(start_node, walk_length) for _ in range(num_walks)]

# DeepWalk
dw = DeepWalk(graph_df)
skip_gram_pairs = dw.generate_skip_gram_data(walks, window_size=2)
embeddings_dw, node_index_dw = dw.train_skip_gram(skip_gram_pairs, embed_size, num_epochs, learning_rate)

# Evaluate DeepWalk embeddings
print("Evaluating DeepWalk embeddings:")
evaluate_embeddings(embeddings_dw, node_index_dw, graph_df)

# Node2Vec
nv = Node2Vec(graph_df, p, q)
biased_walks = [nv.biased_random_walk(start_node, walk_length) for _ in range(num_walks)]
skip_gram_pairs = dw.generate_skip_gram_data(biased_walks, window_size=2)
embeddings_nv, node_index_nv = dw.train_skip_gram(skip_gram_pairs, embed_size, num_epochs, learning_rate)

# Evaluate Node2Vec embeddings
print("Evaluating Node2Vec embeddings:")
evaluate_embeddings(embeddings_nv, node_index_nv, graph_df)

# TripletWalk
tw = TripletWalk(graph_df, p, q)
triplet_walks = [tw.triplet_walk(start_node, walk_length) for _ in range(num_walks)]
skip_gram_pairs = dw.generate_skip_gram_data(triplet_walks, window_size=2)
embeddings_tw, node_index_tw = dw.train_skip_gram(skip_gram_pairs, embed_size, num_epochs, learning_rate)

# Evaluate TripletWalk embeddings
print("Evaluating TripletWalk embeddings:")
evaluate_embeddings(embeddings_tw, node_index_tw, graph_df)

# Compare Embedding Similarities
def compare_embedding_similarity(embeddings, node_index, graph_df):
    connected_similarities = []
    unconnected_similarities = []
    
    for i, node in enumerate(graph_df['node']):
        for j in range(i + 1, len(graph_df['node'])):
            other_node = graph_df['node'].iloc[j]
            similarity = np.dot(embeddings[node_index[node]], embeddings[node_index[other_node]])
            
            if other_node in graph_df[graph_df['node'] == node]['neighbors'].values[0]:
                connected_similarities.append(similarity)
            else:
                unconnected_similarities.append(similarity)
    
    return connected_similarities, unconnected_similarities

# Calculate similarities for DeepWalk embeddings
connected_sim_dw, unconnected_sim_dw = compare_embedding_similarity(embeddings_dw, node_index_dw, graph_df)

# Calculate similarities for Node2Vec embeddings
connected_sim_nv, unconnected_sim_nv = compare_embedding_similarity(embeddings_nv, node_index_nv, graph_df)

# Calculate similarities for TripletWalk embeddings
connected_sim_tw, unconnected_sim_tw = compare_embedding_similarity(embeddings_tw, node_index_tw, graph_df)

# Print results
print("Average similarity for connected nodes (DeepWalk):", np.mean(connected_sim_dw))
print("Average similarity for unconnected nodes (DeepWalk):", np.mean(unconnected_sim_dw))

print("Average similarity for connected nodes (Node2Vec):", np.mean(connected_sim_nv))
print("Average similarity for unconnected nodes (Node2Vec):", np.mean(unconnected_sim_nv))

print("Average similarity for connected nodes (TripletWalk):", np.mean(connected_sim_tw))
print("Average similarity for unconnected nodes (TripletWalk):", np.mean(unconnected_sim_tw))
