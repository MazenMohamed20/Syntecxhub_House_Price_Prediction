# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.87-brightgreen?style=for-the-badge)

> Predicting house sale prices using **Linear Regression** on the Kaggle House Prices dataset.  
> Built as part of the **SyntecxHub Machine Learning Track — Week 1 Project**.

---

## 📌 Project Overview

This project builds a machine learning model that predicts the sale price of houses based on key features like size, quality, location, and condition.

| Metric | Value |
|--------|-------|
| **Model** | Linear Regression |
| **R² Score** | 0.870 |
| **RMSE** | 0.155 (log scale) |
| **Dataset** | Kaggle — House Prices: Advanced Regression Techniques |
| **Training Samples** | 1,168 |
| **Test Samples** | 292 |

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── main.py                  # Main training pipeline
├── predict.py               # Load model & make predictions
├── house_price_model.pkl    # Saved trained model
├── train.csv                # Dataset
└── README.md                # You are here
```

---

## 🔄 Pipeline

```
Load Data → Clean & Handle Missing Values → Feature Selection
    → Train/Test Split → Train Model → Evaluate → Save Model → Predict
```

### Step-by-step:

**1. Load & Explore**
```python
data = pd.read_csv('train.csv')
print(data.shape)        # (1460, 81)
print(data.isnull().sum())
```

**2. Handle Missing Values**
```python
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
```

**3. Feature Selection**
```python
features = data_clear.select_dtypes(include=['int64', 'float64']).columns.tolist()
features.remove('SalePrice')
features.remove('Id')
```

**4. Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**5. Log Transform Target**
```python
y = np.log(data_clear['SalePrice'])   # Reduces RMSE significantly
```

**6. Train & Evaluate**
```python
model = LinearRegression()
model.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
```

---

## 📊 Results

```
RMSE      : 0.155
R² Score  : 0.870
```

The model explains **87% of the variance** in house prices, with a low prediction error after applying log transformation to the target variable.

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 3. Train the model
```bash
python main.py
```

### 4. Make predictions
```bash
python predict.py
```

---

## 🧰 Tech Stack

- **Python 3.11**
- **Pandas** — Data loading & manipulation
- **NumPy** — Numerical operations
- **Scikit-Learn** — ML model & evaluation
- **Matplotlib** — Visualization
- **Pickle** — Model serialization

---

## 📈 Key Insights

- `OverallQual` (Overall Quality) is the strongest predictor of sale price
- `GrLivArea` (Above-ground living area) has a strong positive correlation
- Applying **log transform** to `SalePrice` significantly improved RMSE
- Using all numerical features outperformed manual feature selection

---

## 👤 Author

**Mazen Mohamed**  
Machine Learning Trainee — SyntecxHub  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
