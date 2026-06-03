import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, model_selection
import pickle

data=pd.read_csv(r"E:\SYNTECXHUB\Task_1_House Price Prediction\train.csv")
#print(data.info())
#print(data.shape)
#print(data.head())
#print(data.describe())
#print(data.isnull().sum())

if data.isnull().sum().sum():
    print(f"There are missing values in the dataset.{data.isnull().sum().sum()} missing values in total.")

data["LotFrontage"]=data["LotFrontage"].fillna(data["LotFrontage"].median())
data_clear=data.copy() 
#print(data_clear.isnull().sum()) 

data_clear.to_csv(r"E:\SYNTECXHUB\Task_1_House Price Prediction\data_clear.csv", index=False) # to save the cleaned dataset to a new CSV file

#print(data_clear.shape)

#print(data_clear.select_dtypes(include=["int64","float64"]).columns.tolist())

# Define the features and target variable   
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
            'FullBath', 'YearBuilt', 'YearRemodAdd', 'LotArea',
            'OverallCond', 'Fireplaces', 'BedroomAbvGr', 'TotRmsAbvGrd',]

x = data_clear[features]
y = np.log(data_clear['SalePrice'])  # Apply log transformation to the target variable

#print(x.shape)
#print(y.shape)


# Split the data into training and testing sets
x_train,x_test,y_train,y_test=model_selection.train_test_split(
    x,y,test_size=0.2,random_state=42
    )

#model training
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
#print("Model trained successfully.")

y_pred=model.predict(x_test)
#print("Predictions made successfully.")

# Calculate Root Mean Squared Error (RMSE)
RMSE=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print(f"Root Mean Squared Error: {RMSE}")

# Calculate R-squared
R2=metrics.r2_score(y_test,y_pred)
print(f"R-squared: {R2}")

# Save the model to a file using pickle
with open(r"E:\SYNTECXHUB\Task_1_House Price Prediction\house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully.")

#Root Mean Squared Error: 0.155
#R-squared: 0.871
