import pickle
import numpy as np
# Load the model from the file
with open(r"E:\SYNTECXHUB\Task_1_House Price Prediction\house_price_model.pkl", "rb") as file:
    model = pickle.load(file)
prediction=model.predict([[7, 1710, 2, 856, 2, 2003, 2003, 8450, 5, 0, 3, 8]])
print(f"Predicted Sale Price for the sample: ${np.exp(prediction)[0]:.2f}")