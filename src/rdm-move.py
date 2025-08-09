import numpy as np
import pandas as pd

# Read the Excel file into a Pandas dataframe
file_path = 'reward-27x27.xlsx'  # Replace with the path to your Excel file
sheet_name = 'Sheet1'  # Replace with the name of the sheet containing the matrix
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Convert the dataframe to a numpy array (matrix)
R = df.to_numpy()

# Q matrix
Q = np.zeros(R.shape)

# Number of iterations
n_iterations = 3200000

# Gamma (learning parameter)
gamma = 0.8

# Cumulative reward
cumulative_reward = 0

def available_actions(state):
    return np.where(R[state] >= 0)[0]

def random_action(actions):
    return np.random.choice(actions)

for _ in range(n_iterations):
    # Select a random state
    current_state = np.random.randint(0, R.shape[0])

    # Get available actions for the current state
    actions = available_actions(current_state)

    # Choose a random action
    action = random_action(actions)

    # Get the next state and its reward
    next_state = action
    reward = R[current_state, action]

    # Update the Q value using the Random-selection algorithm
    Q[current_state, action] = reward + gamma * np.max(Q[next_state])
    
    # Update the cumulative reward
    cumulative_reward += reward

# Normalize the Q matrix
Q_normalized = Q / np.max(Q)

print("Q Matrix:")
print(Q_normalized)

# Print the cumulative reward
# print("Cumulative Reward:", cumulative_reward)

# Set initial state and goal state
initial_state = 0
goal_state = 16

# Find the shortest path to the goal
current_state = initial_state
path = [current_state]

max_steps = 10
step_count = 0

while current_state != goal_state and step_count < max_steps:
    actions = available_actions(current_state)
    if len(actions) == 0:
        break
    next_state = random_action(actions)
    path.append(next_state)
    current_state = next_state
    step_count += 1

if current_state == goal_state:
    print("Shortest path to the goal:")
    print(path)
else:
    print("Failed to reach the goal within the maximum number of steps.")
