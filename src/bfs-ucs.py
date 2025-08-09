import pandas as pd
import heapq

# Read the Excel file into a Pandas dataframe
df = pd.read_excel('adj-matrix-ucs.xlsx', sheet_name='Sheet1', na_values='')

# Convert the dataframe to a matrix
matrix = df.values

def ucs(graph, start, goal):
    visited = set()
    queue = [(0, start, [])]

    while queue:
        (cost, current_node, path) = heapq.heappop(queue)
        if current_node not in visited:
            visited.add(current_node)
            path = path + [current_node]

            if current_node == goal:
                return cost, path

            for neighbor, edge_cost in enumerate(graph[current_node]):
                if neighbor not in visited and edge_cost > 0:
                    heapq.heappush(queue, (cost + edge_cost, neighbor, path))

    return "No path found", []

start_node = 0
goal_node = 24

result = ucs(matrix, start_node, goal_node)
print("Cost:", result[0])
print("Path:", result[1])
