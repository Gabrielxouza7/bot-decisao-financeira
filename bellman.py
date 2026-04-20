import numpy as np

def value_iteration(T, R, gamma=0.9, theta=1e-6, max_iter=1000):
    """
    T: array [n_states, n_actions, n_states]
    R: array [n_states, n_actions, n_states]
    Retorna V* e política ótima π*
    """
    n_states, n_actions, _ = T.shape
    V = np.zeros(n_states)
    history = []  # para plotar convergência

    for iteration in range(max_iter):
        V_old = V.copy()

        for s in range(n_states):
            q_values = []
            for a in range(n_actions):
                # Equação de Bellman
                q = np.sum(T[s, a, :] * (R[s, a, :] + gamma * V_old))
                q_values.append(q)
            V[s] = max(q_values)

        delta = np.max(np.abs(V - V_old))
        history.append(delta)

        if delta < theta:
            print(f"Value Iteration convergiu em {iteration+1} iterações.")
            break

    # Extrai política ótima
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = [
            np.sum(T[s, a, :] * (R[s, a, :] + gamma * V))
            for a in range(n_actions)
        ]
        policy[s] = np.argmax(q_values)

    return V, policy, history