import numpy as np

class FinancialMDP:
    """
    Estados: (tendência_mercado, posição_agente)
    tendência: 0 = caindo, 1 = estável, 2 = subindo
    posição:   0 = sem posição, 1 = comprado
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.n_market_states = 3  # caindo, estável, subindo
        self.n_positions = 2   # sem posição, comprado
        self.n_states = self.n_market_states * self.n_positions
        self.n_actions = 3   # 0 = manter, 1 = comprar, 2 = vender

        # Matriz de transição do mercado
        self.market_transition = np.array([
            [0.6, 0.3, 0.1],  # se caindo:  60% cai, 30% estável, 10% sobe
            [0.2, 0.6, 0.2],  # se estável: 20% cai, 60% estável, 20% sobe
            [0.1, 0.3, 0.6],  # se subindo: 10% cai, 30% estável, 60% sobe
        ])

        # Retorno esperado por estado de mercado
        self.price_delta = {0: -1.0, 1: 0.0, 2: +1.0}
        self.transaction_cost = 0.5

    def encode_state(self, market, position):
        return market * self.n_positions + position

    def decode_state(self, state):
        market = state // self.n_positions
        position = state % self.n_positions
        return market, position

    def step(self, state, action):
        """Executa ação e retorna próximo_estado, recompensa, done"""
        market, position = self.decode_state(state)

        reward = 0.0
        new_position = position

        # Lógica de recompensa e mudança de posição
        if action == 1:  # comprar
            if position == 0:
                new_position = 1
                reward -= self.transaction_cost
            # já comprado: ação inválida, sem efeito
        elif action == 2:  # vender
            if position == 1:
                new_position = 0
                reward -= self.transaction_cost

        # Recompensa pela posição no mercado
        if new_position == 1:
            reward += self.price_delta[market]

        # Transição estocástica do mercado
        new_market = np.random.choice(
            self.n_market_states,
            p=self.market_transition[market]
        )

        next_state = self.encode_state(new_market, new_position)
        return next_state, reward, False

    def build_transition_reward_tables(self):
        """
        Constrói T[s,a,s'] e R[s,a,s'] analiticamente para o Bellman
        """
        T = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s in range(self.n_states):
            market, position = self.decode_state(s)
            for a in range(self.n_actions):
                # Determina nova posição e recompensa imediata
                reward_base = 0.0
                new_position = position

                if a == 1 and position == 0:
                    new_position = 1
                    reward_base -= self.transaction_cost
                elif a == 2 and position == 1:
                    new_position = 0
                    reward_base -= self.transaction_cost

                for new_market in range(self.n_market_states):
                    ns = self.encode_state(new_market, new_position)
                    prob = self.market_transition[market][new_market]
                    T[s, a, ns] += prob

                    r = reward_base
                    if new_position == 1:
                        r += self.price_delta[market]
                    R[s, a, ns] = r

        return T, R