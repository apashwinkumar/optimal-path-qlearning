import matplotlib.pyplot as plt

# Cumulative rewards
q_learning_rewards = [2199570.149947519, 2199374.4499474806, 205823.20000063258, 206017.95000063497, 68664.09999990775, 137234.14999979208, 342942.40000231285, 480994.50000400457, 686703.699996574, 274644.0500014759]  # Replace with your actual Q-learning cumulative rewards
random_selection_rewards = [39514, 13174, 26183, 1313, 65784, 79445, 92383, 105954, 132449, 11901]  # Replace with your actual Random-selection cumulative rewards
dijkstra_rewards = [4, 3, 6, 5, 2, 4, 5, 4, 1, 3]  # Replace with your actual Dijkstra cumulative rewards

# Sort the cumulative rewards in descending order
q_learning_rewards = sorted(q_learning_rewards,)
random_selection_rewards = sorted(random_selection_rewards,)
dijkstra_rewards = sorted(dijkstra_rewards,)

# Plotting the cumulative rewards
plt.plot(q_learning_rewards, label='Q-learning')
plt.plot(random_selection_rewards, label='Random-selection')
plt.plot(dijkstra_rewards, label='Dijkstra')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Iterations')
plt.legend()

# Save the plot as a picture
plt.savefig('cumulative_rewards_plot.png')

# Display the plot
plt.show()
