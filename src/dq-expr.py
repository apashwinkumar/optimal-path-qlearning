import numpy as np
import pandas as pd
import random

def read_reward_matrix(file):
    df = pd.read_excel(file)
    return df.to_numpy()

def q_learning(reward_matrix, episodes, alpha, gamma, epsilon, experience_replay_size):
    n_states = reward_matrix.shape[0]
    q_table = np.zeros_like(reward_matrix)

    for episode in range(episodes):
        state = random.randint(0, n_states - 1)

        for _ in range(experience_replay_size):
            action = epsilon_greedy(state, q_table, epsilon)
            next_state = action
            reward = reward_matrix[state, action]

            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state

    return q_table

def epsilon_greedy(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))
    else:
        return random.randint(0, q_table.shape[1] - 1)

def find_path(start_state, q_table, goal_states):
    path = [start_state]
    state = start_state

    while state not in goal_states:
        next_state = np.argmax(q_table[state])
        path.append(next_state)
        state = next_state

    return path

# Parameters
reward_file = "reward-27x27.xlsx"
episodes = 100000
alpha = 0.5
gamma = 0.9
epsilon = 0.8
experience_replay_size = 100
goal_states = [17]

# Read the reward matrix from the Excel file
reward_matrix = read_reward_matrix(reward_file)

# Train the Q-learning algorithm
q_table = q_learning(reward_matrix, episodes, alpha, gamma, epsilon, experience_replay_size)

# Find and print the optimal path
start_state = 0
optimal_path = find_path(start_state, q_table, goal_states)
print("Optimal path:", optimal_path)
