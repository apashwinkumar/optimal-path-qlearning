import numpy as np
import pandas as pd
# import time
# Read the Excel file into a Pandas dataframe
df = pd.read_excel('reward-27x27.xlsx', sheet_name='Sheet1', na_values='')

# Convert the dataframe to a matrix
matrix = df.values

matrix = np.nan_to_num(matrix, nan=-1)  # Replace NaN with -1

# R matrix
R = matrix.astype(float)
# Q1 and Q2 matrices
Q1 = np.matrix(np.zeros([27, 27]))
Q2 = np.matrix(np.zeros([27, 27]))

# Gamma (learning parameter) and alpha (learning rate)
gamma = 0.75
alpha = 0.5

# Initial state. (Usually to be chosen at random)
initial_state = int(matrix[0, 0])

# Reward shaping: Add a small negative reward for each step taken
step_penalty = -0.05
R += step_penalty
np.fill_diagonal(R, 0)

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state]
    av_act = np.where(current_state_row >= 0)[0]
    if len(av_act) == 0:
        # Return a default action if there are no available actions
        return [0]
    else:
        return av_act

# This function chooses at random which action to be performed within the range
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1)[0])
    return next_action

#Cumulative reward
cumulative_reward = 0
# This function updates the Q1 and Q2 matrices according to the path selected and the Double Q-learning algorithm
def update(current_state, action, gamma):
    global cumulative_reward
    
    if np.random.rand() < 0.5:
        max_index = np.where(Q1[action,] == np.max(Q1[action,]))[1]
        
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)
        max_value = Q2[action, max_index]
        
        Q1[current_state, action] = (1 - alpha) * Q1[current_state, action] + \
            alpha * (R[current_state, action] + gamma * max_value)     
        cumulative_reward += R[current_state, action]

    else:
        max_index = np.where(Q2[action,] == np.max(Q2[action,]))[1]
        
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)
        max_value = Q1[action, max_index]
        
        Q2[current_state, action] = (1 - alpha) * Q2[current_state, action] + \
            alpha * (R[current_state, action] + gamma * max_value)
        cumulative_reward += R[current_state, action]


# Epsilon value (exploration probability)
epsilon = 1

# Train over 9000 iterations. (Re-iterate the process above).
for i in range(3200000):
    current_state = np.random.randint(0, int(Q1.shape[0]))
    available_act = available_actions(current_state)

    # Check if there are any available actions
    if len(available_act) == 0:
        continue

    # Generate a random number between 0 and 1
    rand = np.random.rand()

    # Exploit (pick the action with the highest Q-value) with probability 1 - epsilon
    if rand > epsilon:
        action = np.argmax(Q1[current_state, available_act] + Q2[current_state, available_act])
    # Explore (pick a random action) with probability epsilon
    else:
        action = sample_next_action(available_act)

    update(current_state, action, gamma)

# Normalize the "trained" Q1 and Q2 matrices
Q = Q1 + Q2
print("Trained Q matrix:")
print(Q/np.max(Q)*100)
print(Q)

# Set goal state
goal_state = 6

current_state = initial_state
steps = [current_state]

max_steps = 5

visited_states = []

# start_time = time.time()

for num_steps in range(max_steps):
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        # Remove previously visited states
        next_step_index = [index for index in next_step_index if index not in visited_states]
        if len(next_step_index) == 0:
            break
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
        if next_step_index in visited_states:
            break

    steps.append(next_step_index)
    visited_states.append(current_state)
    current_state = next_step_index

    if current_state == goal_state:
        break

# end_time = time.time()  # Record the end time for the testing part
# execution_time = end_time - start_time  # Calculate the execution time for the testing part

# print("Execution time for testing part: {:.6f} seconds".format(execution_time))

# Print selected sequence of steps
if current_state == goal_state:
    print("Reached goal state:", current_state)
    print("Selected path:")
    print(steps)
else:
    print("Finding optimal path....")
    
# Print cumulative reward
print("Cumulative reward:", cumulative_reward)
