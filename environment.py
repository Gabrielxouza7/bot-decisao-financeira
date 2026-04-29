from enum import Enum
import numpy as np

class MarketTendency(Enum):
	FALLING = 0
	STABLE = 1
	RISING = 2

class MarketPositions(Enum):
	NO_POSITION = 0
	BOUGHT = 1

class MarketActions(Enum):
	KEEP = 0
	BUY = 1
	SELL = 2

class FinancialMDP:
	"""
		Estados dependem de tendência e posição\n
		tendência:  0 = caindo, 1 = estável, 2 = subindo\n
		posição:    0 = sem posição, 1 = comprado\n
		ação:       0 = manter, 1 = comprar, 2 = vender\n
	"""

	tendencies = list(MarketTendency)
	positions = list(MarketPositions)
	actions = list(MarketActions)

	def __init__(self, seed=None, market_transition=None):
		if seed:
			np.random.seed(seed)

		self.n_tendencies = len(FinancialMDP.tendencies)
		self.n_positions = len(FinancialMDP.positions)
		self.n_actions = len(FinancialMDP.actions)

		self.n_states = self.n_tendencies * self.n_positions

		if market_transition is None:
			self.market_transition = {
				MarketTendency.FALLING: {
					MarketTendency.FALLING: 0.6,
					MarketTendency.STABLE:  0.3,
					MarketTendency.RISING:  0.1,
				},
				MarketTendency.STABLE: {
					MarketTendency.FALLING: 0.2,
					MarketTendency.STABLE:  0.6,
					MarketTendency.RISING:  0.2,
				},
				MarketTendency.RISING: {
					MarketTendency.FALLING: 0.1,
					MarketTendency.STABLE:  0.3,
					MarketTendency.RISING:  0.6,
				}
			}
		else:
			self.market_transition = market_transition

		self.price_delta = {
			MarketPositions.NO_POSITION: {
				MarketPositions.NO_POSITION: {
					MarketTendency.FALLING: 0.1,
					MarketTendency.STABLE:	0.0,
					MarketTendency.RISING: 	-0.1,
				},
				MarketPositions.BOUGHT: {
					MarketTendency.FALLING: -2.0,
					MarketTendency.STABLE:  0.0,
					MarketTendency.RISING:  1.0,
				}
			},
			MarketPositions.BOUGHT: {
				MarketPositions.NO_POSITION: {
					MarketTendency.FALLING: 2.0,
					MarketTendency.STABLE:  -0.1,
					MarketTendency.RISING:  -0.5,
				},
				MarketPositions.BOUGHT: {
					MarketTendency.FALLING: -2.0,
					MarketTendency.STABLE:  0.0,
					MarketTendency.RISING:  3.0,
				}
			}
		}

		self.transaction_cost = 0.25

	@staticmethod
	def encode_state(market: MarketTendency, position: MarketPositions) -> int:
		return market.value * (len(FinancialMDP.positions)) + position.value

	@staticmethod
	def decode_state(state: int) -> tuple[MarketTendency, MarketPositions]:
		market = state // (len(FinancialMDP.positions))
		position = state % (len(FinancialMDP.positions))
		return MarketTendency(market), MarketPositions(position)

	def valid_actions(self, position: MarketPositions) -> list[int]:
		if position == MarketPositions.NO_POSITION:
			return [MarketActions.KEEP.value, MarketActions.BUY.value]
		else:
			return [MarketActions.KEEP.value, MarketActions.SELL.value]

	def step(self, state: int, action: MarketActions) -> tuple[int, float]:
		"""
			Executa ação e retorna próximo_estado, recompensa
		"""

		market, position = FinancialMDP.decode_state(state)

		reward = 0.0
		new_position = position

		if position == MarketPositions.NO_POSITION and action == MarketActions.BUY:
			new_position = MarketPositions.BOUGHT
			reward -= self.transaction_cost

		elif position == MarketPositions.BOUGHT and action == MarketActions.SELL:
			new_position = MarketPositions.NO_POSITION
			reward -= self.transaction_cost

		new_market = np.random.choice(
			FinancialMDP.tendencies,
			p=[self.market_transition[market][t] for t in FinancialMDP.tendencies]
		)

		reward += self.price_delta[position][new_position][new_market]

		next_state = FinancialMDP.encode_state(new_market, new_position)
		return next_state, reward

	def build_transition_reward_tables(self) -> tuple[np.ndarray, np.ndarray]:
		"""
			Constrói T[s,a,s'] e R[s,a,s'] analiticamente para o Bellman
		"""

		T = np.zeros((self.n_states, self.n_actions, self.n_states))
		R = np.zeros((self.n_states, self.n_actions, self.n_states))

		for market in FinancialMDP.tendencies:
			for position in FinancialMDP.positions:
				state = FinancialMDP.encode_state(market, position)
				valid_actions = list(map(lambda x: MarketActions(x), self.valid_actions(position)))
				for action in valid_actions:
					reward_base = 0.0
					new_position = position

					if position == MarketPositions.NO_POSITION and action == MarketActions.BUY:
						new_position = MarketPositions.BOUGHT
						reward_base -= self.transaction_cost

					elif position == MarketPositions.BOUGHT and action == MarketActions.SELL:
						new_position = MarketPositions.NO_POSITION
						reward_base -= self.transaction_cost

					for new_market in FinancialMDP.tendencies:
						new_state = FinancialMDP.encode_state(new_market, new_position)
						prob = self.market_transition[market][new_market]
						T[state, action.value, new_state] += prob

						r = reward_base
						r += self.price_delta[position][new_position][new_market]
						R[state, action.value, new_state] = r

		return T, R