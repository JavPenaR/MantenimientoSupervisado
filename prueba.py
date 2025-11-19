from pipeline_mantenimiento_v_2 import clasificar_falla

resultado = clasificar_falla(
    nueva_lectura={
        "temperatura_entrada_c": 18.91,
        "temperatura_salida_c": 125.57,
        "presion_entrada_bar": 45.56,
        "presion_salida_bar": 220.76,
        "flujo_masico_kgs": 119.87,
        "vibracion_rms_mms": 2.983,
        "velocidad_rpm": 3226,
        "potencia_kw": 129.72,
        "consumo_energia_kwh": 2.151,
        "humedad_relativa_pct": 82.8,
        "horas_operacion": 2394,
        "delta_temp_c": 106.66,
        "delta_presion_bar": 175.2,
        "temperatura_ambiente_c": 13.19,   # ejemplo genérico
        "modo_operacion": "Alta carga"        # o el valor más común en tu dataset
    },
    modelo_pipeline_path="./salidas/modelo_pipeline.pkl",
    modelo_cluster_path="./salidas/modelo_clusters.pkl",
    scaler_path="./salidas/scaler.pkl",
    columnas_path="./salidas/columnas_sensores_usadas.csv",
    df_dominantes_path="./salidas/sensores_dominantes_por_subtipo.csv"
)

print(resultado)