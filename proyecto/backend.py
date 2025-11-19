from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import threading, time, serial
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque


# --- Configuración del modelo ---
NUM_SENSORES = 6
DATOS_NORMALES = 100
VENTANA_ANOMALIAS = 50  # Considera las últimas 50 lecturas

# --- Variables globales ---
datos_calibracion = [] 
anomalias_recientes = deque(maxlen=VENTANA_ANOMALIAS)
modelo_if = None
total_anomalias = 0

app = FastAPI(title="Backend Arduino con Serial")

origins = [
    "http://localhost:8000",
    "https://roaring-rugelach-82a50b.netlify.app",
    "https://f8a8da7838af.ngrok-free.app",
    "http://ratio-plymouth-eco-international.trycloudflare.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def recibir_datos(nuevos_datos_array):
    """
    Recibe un array de datos de 6 sensores y retorna la probabilidad de fallo.
    El valor retornado es un flotante entre 0.0 y 1.0.
    """
    global datos_calibracion, modelo_if, anomalias_recientes
    nuevos_datos_array[2] = 1 if nuevos_datos_array[2] == "Detectado" else 0
    # Fase de calibración/entrenamiento
    if len(datos_calibracion) < DATOS_NORMALES:
        
        datos_calibracion.append(nuevos_datos_array)
        print(len(datos_calibracion), "datos de calibración recibidos...")
        if len(datos_calibracion) == DATOS_NORMALES:
            print("Entrenando modelo Isolation Forest...")
            modelo_if = IsolationForest(contamination='auto', random_state=42)
            modelo_if.fit(np.array(datos_calibracion))
            print("Entrenamiento completo. ¡Listo para la detección!")
        return 0.0 # Retorna 0.0 durante la fase de calibración
    
    # Detección de anomalías
    anomalia_score = modelo_if.decision_function([nuevos_datos_array])[0]
    
    # Registrar si el dato es anómalo (score < 0)
    es_anomalia = anomalia_score < 0
    anomalias_recientes.append(es_anomalia)
    
    # Calcular la probabilidad de fallo
    total_anomalias = sum(anomalias_recientes)
    if es_anomalia:
        print("Anomalía detectada: Score:", anomalia_score, "probabilidad fallo:", total_anomalias / VENTANA_ANOMALIAS)
    probabilidad_fallo = total_anomalias / VENTANA_ANOMALIAS
    
    # Imprimir para demostración, pero el valor a usar es el retorno

    return probabilidad_fallo



if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return JSONResponse({"detail": "Frontend no encontrado"}, status_code=404)

# -------- Lectura serial en background ---------
last_readings = None  # dict con la última lectura transformada

SERIAL_PORT = "/dev/ttyACM0"  # Ajusta a tu puerto
BAUD = 9600

def parse_line(line):
    # asume CSV: Humedad,Vibracion,Infra,Pot1,Pot2
    parts = [p.strip() for p in line.split(",") if p.strip()!=""]
    if len(parts) != 6:
        return None
    return {
        "Humedad (%)": int(parts[0]),
        "Vibracion": int(parts[1]),
        "Infrarrojo": "Detectado" if parts[2]=="1" else "Libre",
        "Pot1": int(parts[3]),
        "Pot2": int(parts[4]),
        "Pot3": int(parts[5])
    }

def serial_loop():
    
    global last_readings
    try:
        with serial.Serial(SERIAL_PORT, BAUD, timeout=1) as ser:
            time.sleep(2)
            while True:
                try:
                    raw = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not raw:
                        continue
                    if raw.lower().startswith("humedad"):
                        continue  # cabecera
                    readings = parse_line(raw)
                    if readings:
                        last_readings = readings
                        last_readings["Probabilidad fallo"] = recibir_datos([
                            readings["Humedad (%)"],
                            readings["Vibracion"],
                            readings["Infrarrojo"],
                            readings["Pot1"],
                            readings["Pot2"],
                            readings["Pot3"]
                        ])
                except Exception as e:
                    print("Error lectura serial:", e)
    except Exception as e:
        print("No se pudo abrir puerto serie:", e)

# Lanzar hilo al arrancar FastAPI
threading.Thread(target=serial_loop, daemon=True).start()

# -------- Endpoints ---------
@app.get("/predict")
async def predict():
    """
    Devuelve la última lectura del Arduino leída por Serial.
    """
    if last_readings is None:
        return {"detail": "No hay lecturas aún"}
    return {"transformed": last_readings}

@app.get("/resset")
async def resset():
    """
    reinicia el modelo de detección de anomalías.
    """
    last_readings = None
    total_anomalias = 0
    datos_calibracion = []
    return {"detail": "Reinicio simulado (no implementado)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)








