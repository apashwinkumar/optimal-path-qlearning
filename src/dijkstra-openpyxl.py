import heapq
import openpyxl
# import timeit

def dijkstra(adj_matrix, start, goal):
    num_nodes = len(adj_matrix)
    distances = [float('inf')] * num_nodes
    distances[start] = 0
    pq = [(0, start)]
    previous_nodes = [-1] * num_nodes
    cumulative_reward = [0] * num_nodes  # Initialize cumulative reward array
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)

        if curr_dist > distances[curr_node]:
            continue

        for neighbor, weight in enumerate(adj_matrix[curr_node]):
            if weight == 0:
                continue

            new_dist = curr_dist + weight
            new_reward = cumulative_reward[curr_node] + adj_matrix[curr_node][neighbor]  # Calculate new cumulative reward


            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                cumulative_reward[neighbor] = new_reward  # Update cumulative reward
                previous_nodes[neighbor] = curr_node
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, previous_nodes, cumulative_reward

def get_shortest_path(previous_nodes, start, goal):
    path = []
    node = goal

    while node != start:
        if node == -1:
            return None  # if No path exists from start to goal

        path.append(node)
        node = previous_nodes[node]

    path.append(start)
    path.reverse()

    return path

def read_adj_matrix_from_excel(file_path, sheet_name):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook[sheet_name]
    matrix = []
    
    for row in sheet.iter_rows():
        row_data = [cell.value for cell in row]
        matrix.append(row_data)

    return matrix

# path to Excel file and the sheet name
excel_file = 'adj-matrix.xlsx'
sheet_name = 'Sheet1'

adj_matrix = read_adj_matrix_from_excel(excel_file, sheet_name)
start_node = 0
goal_node = 6 # the goal node index


def run_dijkstra():
    distances, previous_nodes, cumulative_reward = dijkstra(adj_matrix, start_node, goal_node)
    return distances, previous_nodes, cumulative_reward


# Measure the execution time of the run_dijkstra function
# execution_time = timeit.timeit(run_dijkstra, number=1)

distances, previous_nodes, cumulative_reward = run_dijkstra()
shortest_path = get_shortest_path(previous_nodes, start_node, goal_node)

print(f"Shortest path from node {start_node} to node {goal_node}: {shortest_path}")
print(f"Shortest distance from node {start_node} to node {goal_node}: {distances[goal_node]}")
# print(f"Cumulative reward along the shortest path: {cumulative_reward[goal_node]}")