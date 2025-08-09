# Optimal Pathfinding with Q-Learning & Double Q-Learning
**A reinforcement learning approach for shortest pathfinding in dynamic environments, using Q-Learning, Double Q-Learning, Reward Shaping, and exploration-exploitation strategies.**

## 📜 Description

This project implements and compares Q-Learning, Double Q-Learning, Dijkstra's Algorithm, and Random Selection to find optimal paths in large grid environments.
Tested on 17×17 and 27×27 grids, Q-Learning showed superior path quality and learning ability, while slightly slower than Dijkstra in execution time.

## 🚀 Features

✅ Reinforcement Learning-based pathfinding

✅ Q-Learning & Double Q-Learning algorithms

✅ Reward Shaping for improved learning

✅ Epsilon-greedy with decaying epsilon for exploration-exploitation balance

✅ Comparative analysis with Dijkstra's Algorithm & Random Selection

✅ Performance metrics: path length, completion time, cumulative rewards

## 📊 Results Summary

| Algorithm     | Avg Path Length | Completion Time (s) | Cumulative Rewards  |
| ------------- | --------------- | ------------------- | ------------------- |
| Q-Learning    | Optimal         | 0.00093 - 0.0097    | 2,197,570 → 205,823 |
| Dijkstra      | Optimal         | 0.000023 - 0.000034 | 6 → 1               |
| Random Select | Suboptimal      | Variable            | 132,449 → 11,901    |

## ⚙️ Installation & Usage
#### Clone the repository

`git clone https://github.com/Vikhorz/optimal-path-qlearning.git
cd optimal-path-qlearning`

#### Make sure you have Python 3.x installed with these dependencies:

`pip install numpy pandas openpyxl`

Note: openpyxl is required for reading Excel .xlsx files.

## How to run
Make sure your reward-27x27.xlsx file is in the root directory (or adjust the path in the script).

#### Run your main script:
`python src/dl-ql.py`

Replace main.py with the actual script filename if different.

## What it does

- Loads the reward matrix from Excel (reward-27x27.xlsx)

- Implements Double Q-Learning on a 27×27 grid

- Runs 3.2 million training iterations with epsilon-greedy exploration

- Prints the trained Q-matrix and finds the optimal path to a goal state

- Outputs cumulative rewards and the selected path

## 📖 Citation

If you use this work in your research, please cite:

### APA style:
`Ismail, A. S., Mohammed, Z. A., Hussain, K. M., & Hassan, H. O. (2023). Finding optimal path using Q-learning and Double Q-learning: A comparative study. International Journal of Applied Mathematics and Computer Science.`
### BibTeX:
```bibtex @article{Ismail2023Qlearning, title={Finding Optimal Path Using Q-learning and Double Q-learning: A Comparative Study}, author={Ismail, Aran Sirwan and Mohammed, Zhiar Ahmed and Hussain, Kozhir Mustafa and Hassan, Hiwa Omer}, year={2023}, journal={International Journal of Applied Mathematics and Computer Science} } ```

## 📝 License
This project is licensed under the MIT License. Feel free to use and modify it.

## 📬 Contact
- Author: Aran Sirwan Ismail

- Email: aran.311195034@uhd.edu.iq

- ResearchGate: [Aran-Sirwan](https://www.researchgate.net/profile/Aran-Sirwan)

