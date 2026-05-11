from environment import (FinancialMDP, MarketTendency, MarketPositions, MarketActions)
import numpy as np

def qlearning(
	env: FinancialMDP,
	n_episodes=2000,
	max_steps=200,
	alpha=0.1,
	alpha_decay=True,
	gamma=0.9,
	epsilon_start=1.0,
	epsilon_end=0.1,
	epsilon_decay=0.9995
) -> tuple[np.ndarray, list, list]:
	"""
		Retorna Q-table, recompensas por episódio e histórico de epsilon
	"""

	Q = np.full((env.n_states, env.n_actions), 0, dtype=float)
	visit_counts = np.zeros((env.n_states, env.n_actions), dtype=int)
	episode_rewards = []
	epsilons = []
	epsilon = epsilon_start

	for _ in range(n_episodes):
		market = MarketTendency(np.random.randint(env.n_tendencies))
		position = MarketPositions(np.random.randint(env.n_positions))
		state = FinancialMDP.encode_state(market, position)
		total_reward = 0.0

		for _ in range(max_steps):
			_, position = FinancialMDP.decode_state(state)

			valid = env.valid_actions(position)
			if np.random.random() < epsilon:
				action = MarketActions(np.random.choice(valid))
			else:
				action = MarketActions(max(valid, key=lambda a: Q[state, a]))

			next_state, reward = env.step(state, action)

			valid_next_actions = env.valid_actions((FinancialMDP.decode_state(next_state))[1])
			best_next = max(Q[next_state, a] for a in valid_next_actions)

			visit_counts[state, action.value] += 1
			new_alpha = (alpha / np.sqrt(visit_counts[state, action.value])) if alpha_decay else alpha

			Q[state, action.value] += new_alpha * (reward + (gamma * best_next) - Q[state, action.value])

			total_reward += reward
			state = next_state

		epsilon = max(epsilon_end, epsilon * epsilon_decay)

		episode_rewards.append(total_reward)
		epsilons.append(epsilon)

	return Q, episode_rewards, epsilons