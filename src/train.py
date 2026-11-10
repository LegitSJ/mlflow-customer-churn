
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

EXPERIMENT_NAME = "customer-churn-prediction"
RUN_NAME = "logistic-regression-threshold-tuned"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25  # Fraction of the remaining data

MAX_ITER = 1000

THRESHOLDS = [0.5, 0.4, 0.3]
SELECTED_THRESHOLD = 0.3

TARGET_COLUMN = "Churn"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"


# --------------------------------------------------
# 2. Load and clean data
# --------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Replace missing values with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    print("Dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

def prepare_features(df):

    X = df.drop(columns=["customerID", TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nNumerical features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    return X, y, numeric_features, categorical_features


# --------------------------------------------------
# 4. Split data
# --------------------------------------------------

def split_data(X, y):

    # Reserve 20% for final testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Reserve 25% of the remaining data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------
# 5. Build preprocessing and model pipeline
# --------------------------------------------------

def build_model(numeric_features, categorical_features):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=MAX_ITER)
            )
        ]
    )

    return model


# --------------------------------------------------
# 6. Generate class predictions using a threshold
# --------------------------------------------------

def apply_threshold(probabilities, threshold):

    predictions = [
        POSITIVE_CLASS if probability >= threshold
        else NEGATIVE_CLASS
        for probability in probabilities
    ]

    return predictions


# --------------------------------------------------
# 7. Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(y_true, y_pred, y_probabilities):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_churn": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),

        "recall_churn": recall_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),

        "f1_churn": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true == POSITIVE_CLASS,
            y_probabilities
        )
    }

    return metrics


# --------------------------------------------------
# 8. Compare classification thresholds
# --------------------------------------------------

def evaluate_thresholds(y_val, y_val_probabilities):

    print("\nThreshold comparison (validation data):")

    results = {}

    for threshold in THRESHOLDS:

        y_val_pred = apply_threshold(
            y_val_probabilities,
            threshold
        )

        metrics = calculate_metrics(
            y_val,
            y_val_pred,
            y_val_probabilities
        )

        cm = confusion_matrix(
            y_val,
            y_val_pred,
            labels=[NEGATIVE_CLASS, POSITIVE_CLASS]
        )

        tn, fp, fn, tp = cm.ravel()

        results[threshold] = metrics

        print(f"\nThreshold: {threshold:.1f}")

        print(f"Precision: {metrics['precision_churn']:.4f}")
        print(f"Recall: {metrics['recall_churn']:.4f}")
        print(f"F1: {metrics['f1_churn']:.4f}")

        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives: {tn}")
        print(f"False Negatives: {fn}")

    return results


# --------------------------------------------------
# 9. Evaluate final model on test data
# --------------------------------------------------

def evaluate_test_data(model, X_test, y_test):

    # Generate probabilities once
    y_test_probabilities = model.predict_proba(X_test)[:, 1]

    # Apply the threshold selected using validation data
    y_test_pred = apply_threshold(
        y_test_probabilities,
        SELECTED_THRESHOLD
    )

    metrics = calculate_metrics(
        y_test,
        y_test_pred,
        y_test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS]
    )

    print("\nFinal Test Results")
    print("Selected threshold:", SELECTED_THRESHOLD)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
            zero_division=0
        )
    )

    print("\nTest Metrics:")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics


# --------------------------------------------------
# 10. Main training workflow
# --------------------------------------------------

def main():

    # Load and prepare data
    df = load_data()

    X, y, numeric_features, categorical_features = prepare_features(df)

    # Split data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    # Build model
    model = build_model(
        numeric_features,
        categorical_features
    )

    # Configure MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):

        # Train model using training data only
        model.fit(X_train, y_train)

        print("\nModel training completed.")

        # Generate validation probabilities
        y_val_probabilities = model.predict_proba(X_val)[:, 1]

        # Compare thresholds using validation data
        validation_results = evaluate_thresholds(
            y_val,
            y_val_probabilities
        )

        # Retrieve results for the selected threshold
        selected_validation_metrics = validation_results[
            SELECTED_THRESHOLD
        ]

        # Evaluate once on the untouched test data
        test_metrics = evaluate_test_data(
            model,
            X_test,
            y_test
        )

        # Calculate actual split proportions
        train_fraction = (1 - TEST_SIZE) * (1 - VALIDATION_SIZE)
        validation_fraction = (1 - TEST_SIZE) * VALIDATION_SIZE

        # Log experiment parameters
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": TEST_SIZE,
            "selected_threshold": SELECTED_THRESHOLD,
            "threshold_candidates": ",".join(
                str(t) for t in THRESHOLDS
            ),
            "scaler": "StandardScaler",
            "encoder": "OneHotEncoder"
        })

        # Log selected threshold's validation metrics
        mlflow.log_metrics({
            f"validation_{name}": value
            for name, value in selected_validation_metrics.items()
        })

        # Log final test metrics
        mlflow.log_metrics({
            f"test_{name}": value
            for name, value in test_metrics.items()
        })

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="customer-churn-classifier"
        )

        print("\nMLflow logging completed!")
        print("Selected threshold:", SELECTED_THRESHOLD)


if __name__ == "__main__":
    main()
