import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a quick model (Fixing the test_size keyword argument)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save it in the exact dictionary format your app.py expects
model_info = {"model": model, "classes": ["Setosa", "Versicolor", "Virginica"]}

with open("best_model.pkl", "wb") as f:
    pickle.dump(model_info, f)

print("New Iris model saved successfully as best_model.pkl!")