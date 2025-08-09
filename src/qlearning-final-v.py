import numpy as np
import pandas as pd

# Read the Excel file into a Pandas dataframe
df = pd.read_excel('reward-27x27.xlsx', sheet_name='Sheet1', na_values='')

# Convert the dataframe to a matrix
matrix = df.values

matrix = np.nan_to_num(matrix, nan=-1)  # Replace NaN with -1

# R matrix
R = matrix.astype(float)
print(R)
# Q matrix
Q = np.matrix(np.zeros([27, 27]))

# Gamma (learning parameter).
gamma = 0.85
alpha = 0.5

# Initial state. (Usually to be chosen at random)
initial_state = int(matrix[1, 0])

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state]
    av_act = np.where(current_state_row >= 0)[0]
    if len(av_act) == 0:
        # Return a default action if there are no available actions
        return [0]
    else:
        return av_act

# Get available actions in the current state
available_act = available_actions(initial_state)

# This function chooses at random which action to be performed within the range
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1)[0])
    return next_action

# This function updates the Q matrix according to the path selected and the Q
# learning algorithm
def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    # Q-learning formula
    Q[current_state, action] = (1-alpha)*Q[current_state, action] + \
        alpha*(R[current_state, action] + gamma * max_value)

# Update Q matrix
update(initial_state, available_act[0], gamma)

# Epsilon value (exploration probability)
epsilon = 0.9

# Train over 9000 iterations. (Re-iterate the process above).
for i in range(500000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)

    # Check if there are any available actions
    if len(available_act) == 0:
        continue

    # Generate a random number between 0 and 1
    rand = np.random.rand()

    # Exploit (pick the action with the highest Q-value) with probability 1 - epsilon
    if rand > epsilon:
        action = np.argmax(Q[current_state, available_act])
    # Explore (pick a random action) with probability epsilon
    else:
        action = sample_next_action(available_act)

    update(current_state, action, gamma)


# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q/np.max(Q)*100)
print(Q)

# Set goal state
goal_state = 7

current_state = initial_state
steps = [current_state]

max_steps = 26
for num_steps in range(max_steps):
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

    if current_state == goal_state:
        break

# Print selected sequence of steps
if current_state == goal_state:
    print("Reached goal state:", current_state)
    print("Selected path:")
    print(steps)
else:
    print("Failed to reach the goal.")
