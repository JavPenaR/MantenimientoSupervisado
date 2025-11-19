#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from pipeline_mantenimiento_v_2 import clasificar_falla

# ===== Rutas de modelos =====
MODELO_PIPELINE = "salidas/modelo_pipeline.pkl"
MODELO_CLUSTER = "salidas/modelo_clusters.pkl"
SCALER = "salidas/scaler.pkl"
COLUMNAS = "salidas/columnas_sensores_usadas.csv"
DOMINANTES = "salidas/sensores_dominantes_por_subtipo.csv"

# ===== FastAPI =====
app = FastAPI(
    title="API Mantenimiento Predictivo",
    description="Clasificación de fallas + Front React servido desde FastAPI",
    version="1.0"
)

# ===== Último resultado procesado =====
ULTIMO_RESULTADO = None

# ============================
#   PREDICCIÓN
# ============================

from pydantic import RootModel

class Lectura(RootModel[dict]):
    @property
    def to_dict(self):
        return self.root

@app.post("/predict")
def predecir_lectura(lectura: Lectura):
    global ULTIMO_RESULTADO

    data = lectura.to_dict()

    resultado = clasificar_falla(
        nueva_lectura=data,
        modelo_pipeline_path=MODELO_PIPELINE,
        modelo_cluster_path=MODELO_CLUSTER,
        scaler_path=SCALER,
        columnas_path=COLUMNAS,
        df_dominantes_path=DOMINANTES
    )

    ULTIMO_RESULTADO = {
        "lectura_original": data,
        "resultado": resultado
    }

    return {"status": "ok", "prediccion": resultado}


@app.get("/stream")
def obtener_ultima_lectura():
    if ULTIMO_RESULTADO is None:
        return {"status": "esperando_datos"}

    return ULTIMO_RESULTADO

# ============================
#   SERVIR FRONTEND REACT
# ============================

FRONT_FOLDER = "front/dist"

if os.path.exists(FRONT_FOLDER):
    # Sirve archivos estáticos
    app.mount("/assets", StaticFiles(directory=f"{FRONT_FOLDER}/assets"), name="assets")

    @app.get("/")
    def servir_home():
        return FileResponse(f"{FRONT_FOLDER}/index.html")

    @app.get("/{full_path:path}")
    def servir_spa(full_path: str):
        """
        Permite React Router (SPA).
        Si el archivo no existe, devuelve index.html.
        """
        requested = f"{FRONT_FOLDER}/{full_path}"
        if os.path.isfile(requested):
            return FileResponse(requested)
        return FileResponse(f"{FRONT_FOLDER}/index.html")
