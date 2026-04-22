import pandas as pd
import yfinance as yf
from environment import MarketTendency

def download_data(ticker: str, start: str, end: str, interval='1d') -> pd.DataFrame:
    """
    Baixa dados históricos de um ativo.
    ticker: ex: 'PETR4.SA', 'VALE3.SA', 'AAPL'
    start/end: ex: '2020-01-01'
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    df = df[['Close']].dropna()
    df['return'] = df['Close'].pct_change()
    df = df.dropna()
    return df

def discretize_tendency(returns: pd.Series, lower=-0.01, upper=0.01) -> list[MarketTendency]:
    """
    Converte retornos contínuos em tendências discretas.
    < lower  → FALLING
    > upper  → RISING
    entre    → STABLE
    
    lower/upper são os thresholds (padrão ±1%)
    """
    tendencies = []
    for r in returns:
        if r < lower:
            tendencies.append(MarketTendency.FALLING)
        elif r > upper:
            tendencies.append(MarketTendency.RISING)
        else:
            tendencies.append(MarketTendency.STABLE)
    return tendencies

def estimate_transition_matrix(tendencies: list[MarketTendency]) -> dict:
    """
    Estima a matriz de transição empírica a partir das tendências observadas.
    Conta quantas vezes cada tendência foi seguida por cada outra.
    """
    states = list(MarketTendency)
    
    # Inicializa contagens com 1 (suavização de Laplace: evita prob=0)
    counts = {
        s: {t: 1 for t in states}
        for s in states
    }
    
    for i in range(len(tendencies) - 1):
        current = tendencies[i]
        next_ = tendencies[i + 1]
        counts[current][next_] += 1
    
    # Normaliza para probabilidades
    transition = {}
    for s in states:
        total = sum(counts[s].values())
        transition[s] = {t: counts[s][t] / total for t in states}
    
    return transition

def build_from_ticker(
    ticker: str,
    start='2018-01-01',
    end='2024-01-01',
    lower=-0.01,
    upper=0.01
) -> tuple[dict, pd.DataFrame]:
    """
    Pipeline completo: baixa dados, discretiza e estima transição.
    Retorna (transition_matrix, dataframe_com_retornos)
    """
    print(f"Baixando dados de {ticker}...")
    df = download_data(ticker, start=start, end=end)
    
    tendencies = discretize_tendency(df['return'], lower=lower, upper=upper)
    df['tendency'] = [t.name for t in tendencies]
    
    transition = estimate_transition_matrix(tendencies)
    
    # Mostra distribuição das tendências
    from collections import Counter
    counts = Counter(t.name for t in tendencies)
    total = len(tendencies)
    print(f"\nDistribuição das tendências ({ticker}):")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} dias ({100*count/total:.1f}%)")
    
    print(f"\nMatriz de transição estimada:")
    for s in MarketTendency:
        row = transition[s]
        print(f"  {s.name}: ", {t.name: f"{v:.2f}" for t, v in row.items()})
    
    return transition