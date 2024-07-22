import pandas as pd
import sys

# Check if the number of edges parameter is provided
if len(sys.argv) < 2:
    print("Please provide the number of edges to consider.")
    sys.exit(1)

num_edges = int(sys.argv[1])

# Load the data
file_path = '9606.protein.links.v12.0.txt'
data = pd.read_csv(file_path, sep=' ')

# Select only the specified number of edges
data = data.head(num_edges)

# Process the data to create an adjacency list format
graph_data = {}

for _, row in data.iterrows():
    protein1 = row['protein1']
    protein2 = row['protein2']

    if protein1 not in graph_data:
        graph_data[protein1] = []
    if protein2 not in graph_data:
        graph_data[protein2] = []

    graph_data[protein1].append(protein2)
    graph_data[protein2].append(protein1)

# Convert the adjacency list to a DataFrame
nodes = []
neighbors = []

for node, neighbors_list in graph_data.items():
    nodes.append(node)
    neighbors.append(';'.join(neighbors_list))

graph_df = pd.DataFrame({'node': nodes, 'neighbors': neighbors})

# Save the DataFrame to a CSV file
output_path = 'data/graph_data.csv'
graph_df.to_csv(output_path, index=False)

print(f"Data processed and saved to {output_path}")
