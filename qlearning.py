import numpy as np

def qlearning(env, n_episodes=2000, max_steps=200,
              alpha=0.1, gamma=0.9,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
    """
    Retorna Q-table, recompensas por episódio e histórico de epsilon
    """
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []
    epsilons = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        # Inicialização padronizada: estado aleatório
        state = np.random.randint(env.n_states)
        total_reward = 0.0

        for _ in range(max_steps):
            # Estratégia ε-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions) 
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(state, action)

            # Atualização Q-learning
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + gamma * best_next - Q[state, action]
            )

            total_reward += reward
            state = next_state

            if done:
                break

        # Decaimento de epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        epsilons.append(epsilon)

    return Q, episode_rewards, epsilons