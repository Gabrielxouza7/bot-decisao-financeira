from environment import (FinancialMDP, MarketTendency, MarketPositions, MarketActions)
import numpy as np

def qlearning(
	env: FinancialMDP,
	n_episodes=2000,
	max_steps=200,
	alpha=0.1,
	gamma=0.9,
	epsilon_start=1.0,
	epsilon_end=0.05,
	epsilon_decay=0.995
) -> tuple[np.ndarray, list, list]:
	"""
		Retorna Q-table, recompensas por episódio e histórico de epsilon
	"""

	Q = np.zeros((env.n_states, env.n_actions))
	episode_rewards = []
	epsilons = []
	epsilon = epsilon_start

	for _ in range(n_episodes):
		market = MarketTendency(np.random.randint(env.n_tendencies))
		position = MarketPositions(np.random.randint(env.n_positions))
		state = env.encode_state(market, position)
		total_reward = 0.0

		for _ in range(max_steps):
			if np.random.random() < epsilon:
				action = MarketActions(np.random.randint(env.n_actions))
			else:
				action = MarketActions(np.argmax(Q[state]))

			next_state, reward = env.step(state, action)

			best_next = np.max(Q[next_state])
			Q[state, action.value] += alpha * (reward + (gamma * best_next) - Q[state, action.value])

			total_reward += reward
			state = next_state

		epsilon = max(epsilon_end, epsilon * epsilon_decay)

		episode_rewards.append(total_reward)
		epsilons.append(epsilon)

	return Q, episode_rewards, epsilons