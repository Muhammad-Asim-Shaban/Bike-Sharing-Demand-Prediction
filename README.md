# 🚴‍♂️ Bike Sharing Demand Prediction 📊
This project predicts the number of rented bikes based on environmental and seasonal data using machine learning techniques.

# 📂 Project Structure
Bike_sharing.ipynb – Jupyter notebook containing the full project code

data.csv – Dataset used for training & evaluation

# 🧰 Libraries & Tools
Python 🐍

Pandas & NumPy – Data manipulation

Seaborn – Data visualization

Scikit-learn – Machine learning

# 📊 Dataset Overview
The dataset contains records of hourly bike rentals in Seoul, South Korea. Key features:

Temperature 🌡️

Humidity 💧

Wind speed 💨

Visibility 👀

Solar radiation ☀️

Rainfall 🌧️

Snowfall ❄️

Seasons (Winter, Spring, Summer, Autumn)

Holiday and Functioning Day flags

🎯 Target variable: Rented Bike Count

# 🧹 Data Preprocessing
Handled categorical features using map() and get_dummies():

```python
df['Holiday'] = df['Holiday'].map({'No Holiday': 0, 'Holiday': 1})
df['Functioning Day'] = df['Functioning Day'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['Seasons'], drop_first=True)
```
Standardized numerical columns using StandardScaler():

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```
Dropped non-useful columns:
```python
df.drop('Date', axis=1, inplace=True)
```
# 📈 Exploratory Data Analysis
Used a heatmap to visualize feature correlation:
```python
sns.heatmap(df.corr(), cmap='coolwarm')
```
# 🧠 Model Building
Train-test split:
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Rented Bike Count'])
y = df['Rented Bike Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Model training using Random Forest Regressor:
```python
from sklearn.linear_model import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
```
