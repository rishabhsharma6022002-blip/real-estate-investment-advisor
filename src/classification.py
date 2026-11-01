import os
import joblib
import mlflow

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/feature_engineered_data.csv"
)

target = "Good_Investment"

excluded_columns = [
    "ID",
    "Good_Investment",
    "Future_Price_5Y_Lakhs",
    "Price_per_SqFt",
    "Price_per_SqFt_Lakhs"
]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df[target]


# ==========================================
# 2. IDENTIFY FEATURES
# ==========================================

numerical_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()


# ==========================================
# 3. PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10
    ))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================
# 4. RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 6. CREATE MLFLOW EXPERIMENT
# ==========================================

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(
    "Real_Estate_Investment_Advisor"
)

with mlflow.start_run(
    run_name="Random_Forest_Classifier"
):

    print("\nTraining Random Forest Classifier...")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # ======================================
    # 7. MODEL EVALUATION
    # ======================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_probability
    )

    cm = confusion_matrix(y_test, y_pred)

    print("\n========== CLASSIFICATION RESULTS ==========")

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nROC-AUC:", roc_auc)

    # ======================================
    # 8. LOG PARAMETERS IN MLFLOW
    # ======================================

    mlflow.log_params({
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "random_state": 42,
        "class_weight": "balanced",
        "test_size": 0.20,
        "training_records": len(X_train),
        "testing_records": len(X_test)
    })

    # ======================================
    # 9. LOG METRICS IN MLFLOW
    # ======================================

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    })

    # ======================================
    # 10. SAVE MODEL
    # ======================================

    os.makedirs("models", exist_ok=True)

    model_path = "models/classification_model.pkl"

    joblib.dump(model, model_path)

    # Save model file as MLflow artifact
    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print("\nModel saved successfully!")

    print("\nMLflow experiment tracking completed!")

    print("Accuracy logged:", accuracy)
    print("ROC-AUC logged:", roc_auc)

    print("MLflow Run ID:", mlflow.active_run().info.run_id)
