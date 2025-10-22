import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from models import LoanData

MODEL_PATH = 'gradient_model.pkl'
model = None

app = FastAPI()


@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

@app.post("/predict")
def predict_loan(data: LoanData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        # Convert LoanData to DataFrame
        input_df = pd.DataFrame([data.dict()])
        print(input_df)
        prediction = model.predict(input_df)[0]
        print(prediction)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Błąd 404: Nie znaleziono pliku index.html</h1>"
                    "<p>Upewnij się, że plik index.html znajduje się w tym samym folderze co main.py.</p>",
            status_code=404
        )
