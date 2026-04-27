from environment import FinancialMDP
import numpy as np

def value_iteration(
	env: FinancialMDP,
	T: np.ndarray,
	R: np.ndarray,
	gamma=0.9,
	theta=1e-6,
	max_iter=1000
) -> tuple[np.ndarray, np.ndarray, list, int]:
	"""
		T: array [n_states, n_actions, n_states]
		R: array [n_states, n_actions, n_states]
		Retorna V*, política ótima π*, histórico de deltas e número de iterações
	"""

	n_states = env.n_states
	n_actions = env.n_actions

	V = np.zeros(n_states)
	history = []

	for iteration in range(max_iter):
		V_old = V.copy()

		for state in range(n_states):
			q_values = np.full(n_actions, -np.inf)
			for action in env.valid_actions(env.decode_state(state)[1]):
				q_values[action] = np.sum(T[state, action, :] * (R[state, action, :] + gamma * V))
			V[state] = max(q_values)

		delta = np.max(np.abs(V - V_old))
		history.append(delta)

		if delta < theta:
			print(f'Value Iteration convergiu em {iteration+1} iterações.')
			n_iter = iteration + 1
			break

	policy = np.zeros(n_states, dtype=int)
	for state in range(n_states):
		q_values = np.full(n_actions, -np.inf)
		for action in env.valid_actions(env.decode_state(state)[1]):
			q_values[action] = np.sum(T[state, action, :] * (R[state, action, :] + gamma * V))
		policy[state] = np.argmax(q_values)

	return V, policy, history, n_iter
