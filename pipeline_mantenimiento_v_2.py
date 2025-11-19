#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Mantenimiento Predictivo - Versión 2 (Normalizado -1 a 1)
----------------------------------------------------------------------
- Clasificación de fallas (RandomForestClassifier)
- Descubrimiento automático de subtipos de falla (MiniBatchKMeans con selección automática de K)
- Identificación de sensores dominantes por subtipo
- Generación de nombres de fallas explicables
- Clasificación de nuevas lecturas con porcentaje de similitud
- Normalización de todas las variables numéricas en rango [-1, 1]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import MiniBatchKMeans


# === UTILIDADES ===

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_barh_series(series: pd.Series, title: str, xlabel: str, outfile: str):
    plt.figure(figsize=(8, 5))
    series.iloc[::-1].plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def plot_heatmap(dataframe: pd.DataFrame, title: str, outfile: str):
    data = dataframe.values
    fig, ax = plt.subplots(figsize=(min(12, 2 + 0.4 * dataframe.shape[1]), 0.8 * dataframe.shape[0] + 3))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(dataframe.shape[1]))
    ax.set_xticklabels([textwrap.shorten(str(c), width=18) for c in dataframe.columns], rotation=90)
    ax.set_yticks(np.arange(dataframe.shape[0]))
    ax.set_yticklabels([str(i) for i in dataframe.index])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


# === PIPELINE PRINCIPAL ===

def main(args):
    csv_path = args.csv
    outdir = args.outdir
    ensure_dir(outdir)

    # === Cargar dataset ===
    df = pd.read_csv(csv_path)
    target_col = "fallo"
    drop_cols = [c for c in ["timestamp", "machine_id"] if c in df.columns]

    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c != target_col and c not in drop_cols]
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in [target_col] + drop_cols]

    # === Clasificación binaria ===
    X = df.drop([target_col] + drop_cols, axis=1)
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Compatibilidad sklearn
    import sklearn
    ohe_params = {"handle_unknown": "ignore"}
    if float(sklearn.__version__[:3]) < 1.4:
        ohe_params["sparse"] = False
    else:
        ohe_params["sparse_output"] = False

    # === NORMALIZACIÓN DE DATOS NUMÉRICOS EN RANGO [-1, 1] ===
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**ohe_params), categorical_cols),
            ("num", MinMaxScaler(feature_range=(-1, 1)), numeric_cols),
        ],
        remainder="drop",
    )

    clf = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
    ])

    clf.fit(X_train, y_train)

    # === Reporte ===
    y_pred = clf.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])

    report.to_csv(os.path.join(outdir, "reporte_clasificador_fallo.csv"))
    cm.to_csv(os.path.join(outdir, "matriz_confusion_fallo.csv"))

    # === Importancias ===
    prep_fitted = clf.named_steps["prep"]
    rf_model = clf.named_steps["rf"]

    final_features = []
    if categorical_cols:
        cat_out = prep_fitted.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
        final_features.extend(cat_out)
    final_features.extend(numeric_cols)

    fi = pd.Series(rf_model.feature_importances_, index=final_features).sort_values(ascending=False)
    fi.to_csv(os.path.join(outdir, "importancia_global_clasificador.csv"))
    plot_barh_series(fi.head(15), "Top 15 características más importantes (clasificador FALLA)", "Importancia", os.path.join(outdir, "top_importancias_clasificador_fallo.png"))

    # === Clustering de fallas (solo filas con fallo = 1) ===
    df_fallas = df[df[target_col] == 1].copy()
    num_for_clust = [c for c in numeric_cols if c in df_fallas.columns]

    # Normalización para clustering en [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Xf_scaled = scaler.fit_transform(df_fallas[num_for_clust])

    possible_k = range(2, 100)
    best_k, best_score, best_labels = None, -1, None

    for k in possible_k:
        mk = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=10)
        labels = mk.fit_predict(Xf_scaled)
        try:
            score = silhouette_score(Xf_scaled, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score, best_labels, best_model = k, score, labels, mk

    df_fallas["subtipo_falla"] = best_labels
    best_model.n_clusters = best_k

    # === Sensores dominantes ===
    group_means = df_fallas.groupby("subtipo_falla")[num_for_clust].mean()
    global_means = df_fallas[num_for_clust].mean()
    global_stds = df_fallas[num_for_clust].std().replace(0, np.nan)
    z_diff = (group_means - global_means) / global_stds

    rows = []
    for cl in z_diff.index:
        ranked = z_diff.loc[cl].abs().sort_values(ascending=False)
        for feat, val in ranked.head(10).items():
            rows.append({"subtipo_falla": int(cl), "feature": feat, "abs_z_diff": float(val)})

    dominance_df = pd.DataFrame(rows).sort_values(["subtipo_falla", "abs_z_diff"], ascending=[True, False])
    dominance_df.to_csv(os.path.join(outdir, "sensores_dominantes_por_subtipo.csv"), index=False)
    group_means.to_csv(os.path.join(outdir, "resumen_medias_por_subtipo.csv"))

    plot_heatmap(group_means, f"Medias por subtipo de falla (K={best_k})", os.path.join(outdir, "heatmap_medias_por_subtipo.png"))

    # === Guardar modelos ===
    joblib.dump(clf, os.path.join(outdir, "modelo_pipeline.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    joblib.dump(best_model, os.path.join(outdir, "modelo_clusters.pkl"))
    pd.Series(num_for_clust).to_csv(
        os.path.join(outdir, "columnas_sensores_usadas.csv"),
        index=False,
        header=False
    )

    print(f"✔️ Pipeline completado. K óptimo detectado = {best_k} (silhouette={best_score:.3f})")


# === FUNCIÓN PARA CLASIFICAR NUEVAS LECTURAS ===

def clasificar_falla(nueva_lectura: dict, modelo_pipeline_path, modelo_cluster_path, scaler_path, columnas_path, df_dominantes_path):

    clf = joblib.load(modelo_pipeline_path)
    mk_model = joblib.load(modelo_cluster_path)
    scaler = joblib.load(scaler_path)

    columnas = pd.read_csv(columnas_path, header=None)[0].tolist()
    dominantes = pd.read_csv(df_dominantes_path)

    X = pd.DataFrame([nueva_lectura]).reset_index(drop=True)

    # Clasificador principal
    pred_falla = clf.predict(X)[0]
    if pred_falla == 0:
        return {"es_falla": False, "tipo_falla": "Operación normal", "similitud_pct": 0.0, "subtipo_id": None}

    # Normalización NUMÉRICA para clustering
    Xnum = X[columnas].copy()
    X_scaled = scaler.transform(Xnum)

    distancias = np.linalg.norm(mk_model.cluster_centers_ - X_scaled, axis=1)
    idx_min = np.argmin(distancias)
    min_dist, max_dist = distancias[idx_min], np.max(distancias)
    similitud = 100 * (1 - (min_dist / max_dist))

    top_vars = dominantes[dominantes["subtipo_falla"] == idx_min]["feature"].head(3).tolist()
    tipo_nombre = "Falla_" + "_".join([t.replace("_", "") for t in top_vars])

    return {
        "es_falla": True,
        "tipo_falla": tipo_nombre,
        "similitud_pct": float(similitud),
        "subtipo_id": int(idx_min)
    }


# === EJECUCIÓN ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline mantenimiento predictivo normalizado [-1, 1]")
    parser.add_argument("--csv", type=str, required=True, help="Ruta del CSV con columna 'fallo'")
    parser.add_argument("--outdir", type=str, default="./salidas", help="Directorio de salida")
    args = parser.parse_args()
    main(args)
