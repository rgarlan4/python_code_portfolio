import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

df = pd.read_csv(r'C:\Users\robert\Documents\GitHub\python_code_portfolioMetro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv')

zipcodes = [28202,28208,28203]

df = df[df['RegionName'].isin(zipcodes)]

df = df.melt(id_vars='RegionName', var_name='date',value_name='value')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

df = df[df['year'] >= pd.Timestamp.now().year - 10]

for zipcode in zipcodes:
    df_zip = df[df['RegionName'] == zipcodes]
    plt.plot(df_zip['date'], df_zip['value'], label=f'Zipcode {zipcode}')

for zipcode in zipcodes:
    df_zip = df[df['RegionName'] == zipcode]
    model = LinearRegression()
    X = df_zip['date'].map(datetime.datetime.toordinal).values.reshape(-1,1)
    y = df_zip['values']
    model.fit(X, y)
    X_future = np.array([datetime.datetime.toordinal(pd.Timestamp.now() + pd.DateOffset(years=i)) for i in range (0, 6)]).reshape(-1,1)
    y_future = model.predict(X_future)
    plt.plot([datetime.datetime.fromordinal(int(x)) for x in X_future], y_future, linestyle='dashed', label=f'Predicted for {zipcode}')

plt.xlabel('date')
plt.ylabel('Home Values')
plt.legend()
plt.title('Home values by year')
plt.show()