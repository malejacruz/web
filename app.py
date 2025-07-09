import warnings
# Suprimimos la advertencia de XGBoost por versión antigua
warnings.filterwarnings(
    "ignore", message=r".*older version of XGBoost.*", category=UserWarning
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ← Nuevo
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Esquema de datos de entrada con ejemplo para Swagger UI
def default_example():
    return {
        "Platform": "PS4",
        "Genre": "Action",
        "Publisher": "Sony",
        "Company": "Sony Interactive"
    }

class GameInput(BaseModel):
    Platform: str = Field(..., example="PS4")
    Genre: str = Field(..., example="Action")
    Publisher: str = Field(..., example="Sony")
    Company: str = Field(..., example="Sony Interactive")

    class Config:
        schema_extra = {
            "example": default_example()
        }

# Inicializamos FastAPI
app = FastAPI(
    title="Predictor de Región para Ventas de Videojuegos",
    description="API que recibe Platform, Genre, Publisher y Company para predecir la región con más ventas",
    version="1.0"
)

# 🚨 Agregamos el middleware CORS aquí:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O ["http://127.0.0.1:5500"] si usas Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos modelo y encoder con joblib
model = joblib.load("modelo_region_xgb.pkl")
label_encoder = joblib.load("label_encoder_region.pkl")

# Detectamos nombres de características si el modelo los guarda
if hasattr(model, 'feature_names_in_'):
    model_features = list(model.feature_names_in_)
else:
    model_features = None

@app.post("/predict", summary="Predice la región con más ventas para un videojuego")
def predict_region(data: GameInput):
    # Convertimos input a DataFrame y aplicamos one-hot encoding
    df = pd.DataFrame([data.dict()])
    df_encoded = pd.get_dummies(df)

    # Alineamos columnas con el modelo (si disponemos de los nombres)
    if model_features is not None:
        df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

    # Predicción
    try:
        pred_encoded = model.predict(df_encoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

    # Decodificamos la región
    try:
        pred_region = label_encoder.inverse_transform(pred_encoded)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al decodificar la región: {e}")

    # Construimos mensaje personalizado
    message = f"La región ideal para vender el juego es: {pred_region}"
    return {"message": message}

# Entry point para Run & Debug
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
