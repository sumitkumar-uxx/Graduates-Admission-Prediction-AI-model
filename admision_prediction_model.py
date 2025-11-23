import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file  = pd.read_csv(r"C:\Users\sumit\OneDrive\Documents\admi.csv")
frame = pd.DataFrame(file)
print(frame)

inputs = frame[['GRE Score',
                'TOEFL Score',
                'University Rating',
                'SOP',
                'LOR ',
                'CGPA',
                'Research']]

output = frame["Chance of Admit "]

algo  = XGBRegressor()


x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2, random_state=42)

algo.fit(x_train, y_train)

test_predictions = algo.predict(x_test)



mae  = mean_absolute_error(y_test, test_predictions)
mse  = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, test_predictions)

print("\n--- MODEL TEST RESULTS ---")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


gre =  int(input("Enter your GRE Score:="))
tofel = int(input("Enter your TOEFL score:="))
rating = int(input("Enter your University Rating:="))
sop = float(input("Enter your SOP:="))
lor = float(input("Enter your LOR:="))
Cgpa = float(input("Enter your CGPA:="))
reserch = int(input("Enter your Research yes=1, 0=NO:="))

getter = algo.predict([[gre, tofel, rating, sop, lor, Cgpa, reserch]])

print("Your chance for getting admission:")
print(getter)
