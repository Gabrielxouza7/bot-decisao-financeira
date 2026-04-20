from environment import FinancialMDP
from bellman import value_iteration
from qlearning import qlearning
from evaluate import (plot_learning_curve, plot_value_map,
                      plot_policy, plot_trajectory, compare_gammas)

# Setup
env = FinancialMDP(seed=42)
T, R = env.build_transition_reward_tables()

# Bellman
print("=== Value Iteration ===")
V_star, pi_star, vi_history = value_iteration(T, R, gamma=0.9)
print("V* =", V_star)
print("π* (Bellman) =", pi_star)
plot_value_map(V_star, env.n_market_states, env.n_positions)
plot_policy(pi_star, env.n_market_states, env.n_positions)

# Q-learning
print("\n=== Q-learning ===")
Q, rewards, epsilons = qlearning(env, n_episodes=2000, gamma=0.9)
pi_ql = Q.argmax(axis=1)
print("π* (Q-learning) =", pi_ql)
plot_learning_curve(rewards)
plot_value_map(Q.max(axis=1), env.n_market_states, env.n_positions)
plot_policy(pi_ql, env.n_market_states, env.n_positions)
plot_trajectory(env, pi_ql)

# Análise experimental
print("\n=== Comparação de γ ===")
compare_gammas(env)