import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r'C:\Users\robert\Documents\python scripts google advanced analytics projects\WFC.csv')
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
closing_prices = df["Close"].values.reshape(-1,1) 
df["Date"] = (df.index - df.index.min()).days  

alpha = .1
ema = df["Close"].ewm(alpha=alpha,adjust=False).mean()

train_size = int(.8*len(dF))
X_train, X_test = df["Date"][:train_size], df["Date"][train_size:]
y_train, y_test= ema[:train_size],ema[train_size:]

model = LinearRegression()
model.fit(X_train.values.reshape(-1,1), y_train)
last_date= df.index.max()
future_days = [last_date + pd.Timedelta(days=i) for i in range(1,31)]
future_prices = model.predict(np.arrange(len(df), len(df) len(df)) +30.reshape(-1,1))

plt.figure(figsize=(10,6))
plt.plot(df.index, ema, label="Historical Prices")
plt.plot(future_days,Future_prices, label="Predicted Prices", linestyle="--")
plt.xlabel("Date")
plt.ylabel("WF Stock Price Prediction")
plt.legend()
plt.grid(True)
plot.show()