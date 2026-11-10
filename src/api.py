from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import logging

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field


# Configuration
logger = logging.getLogger(__name__)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "customer-churn-classifier"
)

CHURN_THRESHOLD = 0.3


# Request and response schemas
class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float
    threshold: float
    message: str


# Model loading
def load_packaged_model():
    """Load the packaged MLflow model from the project directory."""

    model = mlflow.sklearn.load_model(str(MODEL_PATH))

    if "Yes" not in model.classes_:
        raise ValueError("The loaded model does not contain the churn class 'Yes'.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support probability predictions.")

    return model, CHURN_THRESHOLD


## Historical alternative: load the latest model directly from MLflow tracking.
## Keep this for reference; it requires importing MlflowClient.
#
#from mlflow.tracking import MlflowClient
# def load_model_locally():
#     client = MlflowClient()
#
#     experiment = client.get_experiment_by_name(
#         "customer-churn-prediction"
#     )
#
#     if experiment is None:
#         raise ValueError("MLflow experiment not found.")
#
#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         filter_string=(
#             "tags.mlflow.runName = 'logistic-regression-threshold-tuned'"
#         ),
#         order_by=["start_time DESC"],
#         max_results=1
#     )
#
#     if not runs:
#         raise ValueError("No trained model runs found.")
#
#     run = runs[0]
#     run_id = run.info.run_id
#
#     model_uri = f"runs:/{run_id}/customer-churn-classifier"
#     model = mlflow.sklearn.load_model(model_uri)
#
#     threshold = float(run.data.params["selected_threshold"])
#
#     return model, threshold


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading packaged MLflow model...")

    app.state.model, app.state.threshold = load_packaged_model()

    logger.info("MLflow model loaded successfully.")

    yield

    logger.info("API shutting down.")


# FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using an MLflow model.",
    version="1.0.0",
    lifespan=lifespan
)


# Routes
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!"
    }


@app.get("/health")
def health(request: Request):
    model_loaded = (
        getattr(request.app.state, "model", None) is not None
        and getattr(request.app.state, "threshold", None) is not None
    )

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData, request: Request):
    try:
        customer_df = pd.DataFrame([customer.model_dump()])

        model = request.app.state.model
        threshold = request.app.state.threshold

        probabilities = model.predict_proba(customer_df)

        churn_index = list(model.classes_).index("Yes")
        churn_probability = probabilities[0][churn_index]

        prediction = (
            "Yes" if churn_probability >= threshold else "No"
        )

        return {
            "prediction": prediction,
            "churn_probability": round(float(churn_probability), 4),
            "threshold": threshold,
            "message": (
                "Customer is predicted to churn."
                if prediction == "Yes"
                else "Customer is predicted to stay."
            )
        }

    except Exception:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the prediction."
        )
