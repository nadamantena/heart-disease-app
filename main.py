# main.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load Dataset
df = pd.read_csv("data/heart.csv")
print("✅ Dataset loaded successfully!")

# 2. Handle missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))

# 3. Encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 4. Convert 'num' to binary target
df_encoded['num'] = (df_encoded['num'] > 0).astype(int)

# 5. Features and target
X = df_encoded.drop(["num", "id"], axis=1)
y = df_encoded["num"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# 9. Tune models and store best
best_models = {}
best_scores = {}

for name in models:
    print(f"\n🔍 Tuning {name} ...")
    grid = GridSearchCV(
        models[name], param_grids[name], cv=5,
        scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    best_scores[name] = grid.best_score_
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

# 10. Evaluate best models
test_results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    test_results[name] = acc
    print(f"\n✅ {name} Test Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 11. Save best model + scaler + column order
best_model_name = max(test_results, key=test_results.get)
best_model = best_models[best_model_name]

pipeline = {
    "model": best_model,
    "scaler": scaler,
    "columns": X.columns.tolist()
}

joblib.dump(pipeline, "heart_disease_pipeline.pkl")
print(f"\n🏆 Best Model: {best_model_name} | Accuracy: {test_results[best_model_name]:.4f}")
print("💾 Saved full pipeline as 'heart_disease_pipeline.pkl'")
