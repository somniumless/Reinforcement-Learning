import pandas as pd
import numpy as np

dates = pd.date_range(start="2023-01-01", periods=100)
prices = np.cumsum(np.random.randn(100) * 10 + 100)

data = pd.DataFrame({
    "date": dates,
    "open": prices,
    "high": prices + np.random.rand(100) * 5,
    "low": prices - np.random.rand(100) * 5,
    "close": prices
})

data.to_csv("data/stock_data.csv", index=False)
print("¡Archivo stock_data.csv generado!")