# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("bank.csv", sep=';')

# Convert categorical data into numeric (BEST METHOD)
df = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (Top Levels)")
plt.show()