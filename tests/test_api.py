from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

test_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "model_loaded": True
        }
        
        
def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=test_customer
        )

        assert response.status_code == 200

        result = response.json()

        assert result["prediction"] in ["Yes", "No"]
        if result["prediction"] == "Yes":
            assert result["message"] == "Customer is predicted to churn."
        else:
            assert result["message"] == "Customer is predicted to stay."
        assert 0 <= result["churn_probability"] <= 1
        assert result["threshold"] == 0.3
        assert "message" in result
        
def test_predict_invalid_gender():
    invalid_customer = test_customer.copy()
    invalid_customer["gender"] = "UNCLE"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_customer
        )

        assert response.status_code == 422

def test_predict_invalid_tenure():
    invalid_customer = test_customer.copy()
    invalid_customer["tenure"] = -1

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_customer
        )

        assert response.status_code == 422


