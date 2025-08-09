import openpyxl
import numpy as np
import random
from typing import List

def load_reward_matrix(filename: str, sheet_name: str) -> List[List[int]]:
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook[sheet_name]

    matrix = []
    for row in sheet.iter_rows():
        matrix.append([cell.value for cell in row])

    return matrix

def epsilon_greedy_action(Q: np.ndarray, state: int, epsilon: float) -> int:
    valid_actions = np.where(Q[state] > -np.inf)[0]
    if valid_actions.size == 0:
        return None  # No valid actions for the given state

    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)
    else:
        return np.argmax(Q[state])

def q_learning(reward_matrix: List[List[int]], source: int, destination: int, episodes: int, alpha: float, gamma: float, epsilon: float):
    n_states = len(reward_matrix)
    Q = np.full((n_states, n_states), -np.inf)
    for state in range(n_states):
        Q[state][[i for i, r in enumerate(reward_matrix[state]) if r >= 0]] = 0

    for _ in range(episodes):
        state = source
        while state != destination:
            action = epsilon_greedy_action(Q, state, epsilon)
            next_state = action
            Q[state][action] = Q[state][action] + alpha * (reward_matrix[state][action] + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state

    return Q


def shortest_path(Q: np.ndarray, source: int, destination: int) -> List[int]:
    path = [source]
    state = source
    while state != destination:
        next_state = np.argmax(Q[state])
        path.append(next_state)
        state = next_state
    return path

def main():
    filename = "reward-27x27-Copy.xlsx"  # Replace with your Excel file name
    sheet_name = "Sheet1"  # Replace with your sheet name
    source_node = 0
    destination_node = 15  # Replace with your destination node
    episodes = 5000
    alpha = 0.5
    gamma = 0.8
    epsilon = 0.85

    reward_matrix = load_reward_matrix(filename, sheet_name)
    Q = q_learning(reward_matrix, source_node, destination_node, episodes, alpha, gamma, epsilon)
    path = shortest_path(Q, source_node, destination_node)

    print(f"Shortest path from node {source_node} to node {destination_node}: {path}")

if __name__ == "__main__":
    main()
