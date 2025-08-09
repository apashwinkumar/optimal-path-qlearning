import pandas as pd
import heapq
import numpy as np
import math

def euclidean_distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

def a_star_search(adj_matrix, start, goal):
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path

    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in range(1, len(adj_matrix) + 1)}
    g_score[start] = 0
    f_score = {node: float('inf') for node in range(1, len(adj_matrix) + 1)}
    f_score[start] = euclidean_distance((start, start), (goal, goal))

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in range(1, len(adj_matrix) + 1):
            if adj_matrix[current - 1][neighbor - 1] > 0:
                tentative_g_score = g_score[current] + adj_matrix[current - 1][neighbor - 1]
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + euclidean_distance((neighbor, neighbor), (goal, goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

filename = 'adj-matrix-star.xlsx'
sheet_name = 'Sheet1'
adj_matrix = pd.read_excel(filename, sheet_name=sheet_name, header=None).values

start_node = 1
goal_node = 18

path = a_star_search(adj_matrix, start_node, goal_node)
print("Path:", path)
