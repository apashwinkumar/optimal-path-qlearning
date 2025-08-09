import pandas as pd
import heapq

# Read the Excel file into a Pandas dataframe
df = pd.read_excel('adj-matrix.xlsx', sheet_name='Sheet1', na_values='0')

# Convert the dataframe to a matrix
reward_matrix = df.values

# Dijkstra algorithm function
def dijkstra(graph, start, end):
    # Initialize variables
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    queue = [(0, start)]
    visited = set()
    path = {}
    
    while queue:
        # Get the node with the minimum distance from the start node
        current_distance, current_vertex = heapq.heappop(queue)

        if current_vertex == end:
            # Build the shortest path
            path_list = [current_vertex]
            while current_vertex in path:
                current_vertex = path[current_vertex]
                path_list.append(current_vertex)
            path_list.reverse()
            return path_list

        if current_vertex in visited:
            continue

        # Visit the current node
        visited.add(current_vertex)

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # Update the distance to the neighbor if it's shorter than the current distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                path[neighbor] = current_vertex
                heapq.heappush(queue, (distance, neighbor))

# Test the Dijkstra algorithm
start_node = 0
goal_node = 19
shortest_path = dijkstra(reward_matrix, start_node, goal_node)

print(f"The shortest path from {start_node} to {goal_node} is: {shortest_path}")
