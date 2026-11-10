from mlflow.tracking import MlflowClient
import mlflow.sklearn
import pandas as pd

# Connect to MLflow tracking
client = MlflowClient()

# Find our MLflow experiment
experiment = client.get_experiment_by_name(
    "customer-churn-prediction"
)

if experiment is None:
    raise ValueError("MLflow experiment not found.")

# Find the latest run with our chosen run name
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'logistic-regression-threshold-tuned'",
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No trained model runs found.")

# Retrieve the latest run
run = runs[0]
run_id = run.info.run_id

# Build the model URI using the retrieved run ID
model_uri = f"runs:/{run_id}/customer-churn-classifier"

# Load the model
loaded_model = mlflow.sklearn.load_model(model_uri)

# Retrieve the threshold from the same MLflow run
selected_threshold = float(run.data.params["selected_threshold"])

print("Selected run ID:", run_id)
print("Selected threshold:", selected_threshold)
print("Model loaded successfully!")
print("Loaded model:", type(loaded_model))

# Load the original dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)

# Select one customer record
customer = df.drop(
    columns=["customerID", "Churn"]
).iloc[[0]]

# Display customer information
print("\nCustomer information:")
print(customer)

# Generate churn probability
probabilities = loaded_model.predict_proba(customer)

# Find the probability column corresponding to churn
churn_index = list(loaded_model.classes_).index("Yes")

churn_probability = probabilities[0][churn_index]

# Apply the threshold retrieved from MLflow
prediction = (
    "Yes"
    if churn_probability >= selected_threshold
    else "No"
)

print("\nPrediction:", prediction)
print("Churn probability:", round(churn_probability * 100, 2), "%")

# Retrieve the customer's actual churn status
actual_churn = df.iloc[0]["Churn"]

print("Actual churn status:", actual_churn)
print("Prediction correct:", prediction == actual_churn)