import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_graph(filepath):
    graph_df = pd.read_csv(filepath)
    graph_df['neighbors'] = graph_df['neighbors'].apply(lambda x: x.split(';'))
    return graph_df

def evaluate_embeddings(embeddings, node_index, graph_df):
    similarity_matrix = cosine_similarity(embeddings)
    nodes = list(node_index.keys())
    
    # Calculate average similarity for connected and unconnected nodes
    connected_similarities = []
    unconnected_similarities = []

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:
                continue
            sim = similarity_matrix[i, j]
            if node2 in graph_df[graph_df['node'] == node1]['neighbors'].values[0]:
                connected_similarities.append(sim)
            else:
                unconnected_similarities.append(sim)

    avg_connected_sim = sum(connected_similarities) / len(connected_similarities)
    avg_unconnected_sim = sum(unconnected_similarities) / len(unconnected_similarities)

    print(f"Average similarity for connected nodes: {avg_connected_sim:.4f}")
    print(f"Average similarity for unconnected nodes: {avg_unconnected_sim:.4f}")

    # Print some example similarities
    # for i in range(len(nodes)):
    #     for j in range(i+1, len(nodes)):
    #         print(f"Similarity between {nodes[i]} and {nodes[j]}: {similarity_matrix[i, j]}")
