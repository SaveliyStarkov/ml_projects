import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from db import save_prediction
from training import train_model

app = FastAPI()

MODEL_PATH = "model.pkl"
MODEL_VERSION = "v1"

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

FEATURE_NAMES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'total_late_payments',
    'DebtRatio',
    'age',
    'NumberOfOpenCreditLinesAndLoans',
    'MonthlyIncome',
    'income_per_dependent',
    'NumberOfTimes90DaysLate'
]


class Features(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0, le=2)
    total_late_payments: int = Field(ge=0, le=30)
    DebtRatio: float = Field(ge=0, le=10)
    age: int = Field(ge=21, lt=105)
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0, le=85)
    MonthlyIncome: float = Field(ge=0, lt=8_000_000)
    income_per_dependent: float = Field(ge=0, lt=4_200_000)
    NumberOfTimes90DaysLate: int = Field(ge=0, le=13)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    shap_values: list
    expected_value: float
    feature_names: list
    feature_values: list


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(data: Features):
    X = np.array([[
        data.RevolvingUtilizationOfUnsecuredLines,
        data.total_late_payments,
        data.DebtRatio,
        data.age,
        data.NumberOfOpenCreditLinesAndLoans,
        data.MonthlyIncome,
        data.income_per_dependent,
        data.NumberOfTimes90DaysLate,
    ]])

    try:
        prediction  = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        shap_vals = explainer.shap_values(X)
        sv = shap_vals[1][0].tolist() if isinstance(shap_vals, list) else shap_vals[0].tolist()

        exp_val = explainer.expected_value
        expected_value = float(exp_val[1] if isinstance(exp_val, (list, np.ndarray)) else exp_val)

        try:
            save_prediction(prediction, probability, MODEL_VERSION)
        except Exception:
            pass  

        return PredictResponse(
            prediction=prediction,
            probability=round(probability, 4),
            shap_values=sv,
            expected_value=expected_value,
            feature_names=FEATURE_NAMES,
            feature_values=X[0].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train():
    global model, explainer
    try:
        result = train_model()
        model    = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))