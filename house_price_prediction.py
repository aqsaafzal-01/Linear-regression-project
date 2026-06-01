import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "Size": [1000, 1500, 1800, 2300, 2800],
    "Price": [20000, 30000, 35000, 45000, 50000]
}

df = pd.DataFrame(data)

X = df[["Size"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[2500]])
print("Predicted Price:", prediction[0])

joblib.dump(model, "house_price_model.pkl")
