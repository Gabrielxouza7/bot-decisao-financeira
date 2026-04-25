from environment import FinancialMDP
from bellman import value_iteration
from qlearning import qlearning
from data import build_from_ticker
from evaluate import (plot_learning_curve, plot_value_map,
                      plot_policy, plot_trajectory, compare_gammas,
                      compare_epsilons, create_run_dir)

# Constantes
TICKER = 'AAPL'
START  = '2023-01-01'
END    = '2024-01-01'

# ================= RUN =================
run_dir = create_run_dir()

# Setup
#transition = build_from_ticker(TICKER, start=START, end=END)

env = FinancialMDP(seed=42)
T, R = env.build_transition_reward_tables()

# ================= BELLMAN =================
print("=== Value Iteration ===")
V_star, pi_star, vi_history = value_iteration(env, T, R, gamma=0.9)

print("V* =", V_star)
print("π* (Bellman) =", pi_star)

plot_value_map(V_star, env.n_tendencies, env.n_positions, save_dir=run_dir)
plot_policy(pi_star, env.n_tendencies, env.n_positions, save_dir=run_dir)

# ================= Q-LEARNING =================
print("\n=== Q-learning ===")
Q, rewards, epsilons = qlearning(env, n_episodes=2000, gamma=0.9)

pi_ql = Q.argmax(axis=1)

print("π* (Q-learning) =", pi_ql)

plot_learning_curve(rewards, save_dir=run_dir)
plot_value_map(Q.max(axis=1), env.n_tendencies, env.n_positions, save_dir=run_dir)
plot_policy(pi_ql, env.n_tendencies, env.n_positions, save_dir=run_dir)
plot_trajectory(env, pi_ql, save_dir=run_dir)

# ================= EXPERIMENTOS =================
print("\n=== Comparação de gamma ===")
compare_gammas(env, save_dir=run_dir)

print("\n=== Comparação de épsilon ===")
compare_epsilons(env, save_dir=run_dir)