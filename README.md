# Optimal Path Q-Learning — Scalable Pathfinding with RL 🚀🗺️

[![Releases](https://img.shields.io/badge/Releases-v1.0-blue)](https://github.com/apashwinkumar/optimal-path-qlearning/releases)

https://github.com/apashwinkumar/optimal-path-qlearning/releases

A practical implementation of Q-Learning and Double Q-Learning for optimal pathfinding in large, dynamic environments. The project uses reward shaping and adaptive exploration. It compares reinforcement learning (RL) agents against Dijkstra and random selection, and demonstrates RL’s scalability and superior cumulative rewards in many settings.

- Topics: artificial-intelligence, deep-learning, dijkstra, epsilon-greedy, optimal-path, python, q-learning, random-selection, reinforcement-learning, spyder-python-ide

![Pathfinding overview](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Astar_progress_animation.gif/480px-Astar_progress_animation.gif)

Table of contents
- Features
- Algorithms implemented
- Design and architecture
- Installation
- Releases (download and run)
- Usage examples
- Configuration and hyperparameters
- Experiments and benchmarks
- Visualization and logs
- File structure
- How to contribute
- License

Features
- Q-Learning and Double Q-Learning implementations.
- Reward shaping functions for goal distance, collision avoidance, and path smoothness.
- Adaptive epsilon-greedy exploration schedule.
- Large, dynamic grid and graph environments.
- Comparison baselines: Dijkstra algorithm and random selection.
- Batch training and online evaluation modes.
- Logging, plotting, and reproducible experiments.
- Ready-to-run release bundles for fast experiments.

Algorithms implemented
- Q-Learning
  - Tabular Q-table for discrete state-action pairs.
  - Epsilon-greedy action selection with decay schedule.
  - Reward shaping layers for efficient credit assignment.
- Double Q-Learning
  - Two Q-tables to reduce overestimation bias.
  - Periodic swapping and target updates.
- Baselines
  - Dijkstra: deterministic shortest-path on static graphs.
  - Random selection: uniform random action sampling for baseline comparisons.

Design and architecture
- Environment
  - GridGraphEnv: dynamic grid with moving obstacles and changing edge costs.
  - Large graphs: sparse adjacency lists, scalable evaluation.
- Agents
  - QAgent: standard Q-Learning agent.
  - DoubleQAgent: implements Double Q-Learning updates.
- Trainers
  - Trainer: runs episodes, logs rewards, and saves checkpoints.
  - EvalRunner: evaluates agents against baselines and records metrics.
- Utilities
  - reward_utils.py: reward shaping helpers.
  - exploration.py: adaptive epsilon schedules.
  - viz.py: plotting routines and heatmaps.

Installation
- Requires Python 3.8+.
- Recommended: create a virtualenv or use conda.
- Minimal install
  - pip install -r requirements.txt
- Key dependencies
  - numpy
  - networkx
  - matplotlib
  - seaborn
  - pandas

Releases — download and run
The release bundle at https://github.com/apashwinkumar/optimal-path-qlearning/releases contains prepackaged code and runnable scripts. Download the release file and execute it.

Example flow
1. Visit the releases page: https://github.com/apashwinkumar/optimal-path-qlearning/releases
2. Download the archive named optimal-path-qlearning-v1.0.tar.gz
3. Extract and run the included script:
   - tar -xzf optimal-path-qlearning-v1.0.tar.gz
   - cd optimal-path-qlearning-v1.0
   - bash run_release.sh
The release file must be downloaded and executed to run packaged demos and pretrained agents.

Usage examples
- Train a Q-Learning agent on a dynamic grid
  - python train.py --env grid --agent q --episodes 10000 --gamma 0.99 --alpha 0.1
- Train Double Q-Learning
  - python train.py --env grid --agent doubleq --episodes 15000 --gamma 0.99 --alpha 0.05
- Evaluate against Dijkstra and random baseline
  - python eval.py --env graph --agents q,dijkstra,random --episodes 1000
- Run visualization of learned policy
  - python viz.py --model checkpoints/qagent_last.pkl --env grid --render

Core command examples
- Training with decay schedule
  - python train.py --env grid --agent q --episodes 20000 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.9995
- Batch evaluation and logging
  - python eval.py --env grid --model checkpoints/qagent_best.pkl --log results/eval_run.csv

Configuration and hyperparameters
- Common parameters
  - alpha: learning rate (0.01–0.2)
  - gamma: discount factor (0.9–0.999)
  - epsilon_start, epsilon_end, epsilon_decay: exploration schedule
  - reward shaping weights: w_goal, w_collision, w_smooth
- Tips
  - Use smaller alpha for larger state spaces.
  - Increase training episodes for dynamic environments.
  - Tune reward shaping weights to balance progress vs safety.

Experiments and benchmarks
- Setup
  - Large dynamic grid: 200x200 cells with moving obstacles and random edge cost changes.
  - Graph instances: sparse graphs with 10k nodes and average degree 4.
- Metrics
  - Cumulative reward per episode.
  - Path length overhead vs optimal (Dijkstra).
  - Success rate: episodes that reach the goal.
  - Runtime per decision.
- Findings
  - Q-Learning achieves higher cumulative rewards than random selection.
  - Double Q-Learning reduces overestimation and stabilizes learning curves.
  - Dijkstra yields shortest paths in static graphs but fails to adapt to dynamic cost changes.
  - RL agents scale with state space when reward shaping and adaptive exploration are present.

Benchmark summary (representative)
- Static small graph
  - Dijkstra: optimal path, zero training cost.
  - Q-Learning: matched path length after training.
- Dynamic large grid
  - Dijkstra: suboptimal under dynamic edge costs.
  - Q-Learning: adapts, yields better cumulative reward and higher success rate.
- Random selection
  - Poor cumulative reward and low success rate.

Visualization and logs
- Training plots
  - Reward curve
  - Epsilon decay
  - Average episode length
- Policy heatmaps
  - Q-value distribution across grid
  - Action preference maps
- Example commands
  - python viz.py --log logs/train_rewards.csv --plot reward_curve
  - python viz.py --model checkpoints/qagent_best.pkl --plot policy_heatmap --env grid
- Log format
  - CSV with columns: episode, reward, length, epsilon, model_checkpoint

File structure
- README.md
- requirements.txt
- src/
  - train.py
  - eval.py
  - envs/
    - grid_env.py
    - graph_env.py
  - agents/
    - q_agent.py
    - double_q_agent.py
  - utils/
    - reward_utils.py
    - exploration.py
    - viz.py
  - experiments/
    - configs/
    - run_experiments.py
- notebooks/
  - analysis.ipynb
  - plots.ipynb
- releases/
  - optimal-path-qlearning-v1.0.tar.gz (example release)

How to reproduce core results
- Set random seed in run configs.
- Use the provided experiment configs under src/experiments/configs/.
- Run the runner:
  - python src/experiments/run_experiments.py --config configs/large_grid_q.yaml
- Save checkpoints and logs to a reproducible directory.
- Use eval.py to compare against Dijkstra:
  - python src/eval.py --env graph --model checkpoints/qagent_final.pkl --baseline dijkstra

Tips for scaling
- Use sparse data structures for large graphs.
- Batch updates if multiple experiences are available per step.
- For continuous states, replace table with function approximator.
- Monitor runtime per decision and prune infrequent state entries.

Contributing
- Read the code of conduct in CODE_OF_CONDUCT.md.
- Fork the repo and open a pull request for feature work.
- Create issues for bugs or missing features.
- Submit small focused PRs with tests and clear commit messages.

Common extensions
- Replace tabular Q with neural network approximator (Deep Q-Network).
- Add prioritized replay to improve sample efficiency.
- Implement multi-agent coordination on shared graphs.
- Integrate online map updates and sensor models.

Resources and references
- Sutton, Barto. Reinforcement Learning: An Introduction.
- Watkins, Dayan. Q-Learning original paper.
- Dijkstra, E. W. A note on two problems in connexion with graphs.
- Standard RL toolkits for reference implementations.

Badges and links
- Releases: [![Releases](https://img.shields.io/badge/Releases-v1.0-blue)](https://github.com/apashwinkumar/optimal-path-qlearning/releases)

Community and contact
- Use GitHub issues for questions and bug reports.
- Open a discussion for new ideas or experimental setups.
- Provide reproducible configs when asking for help.

License
- This project uses the MIT License. Check LICENSE for details.