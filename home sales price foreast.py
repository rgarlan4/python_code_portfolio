import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the data
df = pd.read_csv('Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv', usecols=['RegionID',"2023-03-31",'2023-04-30','2023-05-31','2023-06-30','2023-07-31','2023-08-31','2023-09-30','2023-10-31','2023-11-30','2023-12-31','2024-01-31','2024-02-29','2024-03-31'])

# Reshape the data
X = df.drop('2024-03-31', axis=1)
y = df['2024-03-31']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Home Sales Prices vs Predicted Prices')
plt.show()
