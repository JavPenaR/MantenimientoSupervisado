
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Mantenimiento Predictivo
------------------------------------
Genera:
1) Clasificador binario de FALLA (0/1) + métricas y ranking de importancia
2) Descubrimiento de SUBTIPOS de FALLA (clustering sobre registros con fallo=1)
3) Sensores dominantes por subtipo (ranking por |z-diff| vs media global de fallas)
4) Gráficos y CSVs exportados

Uso:
-----
python pipeline_mantenimiento.py --csv /ruta/datos.csv --outdir ./salidas --k 3

Requisitos:
-----------
pip install pandas numpy scikit-learn matplotlib

Notas:
------
- Para acelerar y mantener robusto el clustering, se usa solo variables numéricas
  y MiniBatchKMeans con K fijo (parámetro --k). Puedes cambiarlo o extender con
  estimación automática de K por métricas (silhouette/CH) si lo necesitas.
"""

import argparse
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import MiniBatchKMeans


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
    fig, ax = plt.subplots(figsize=(min(12, 2 + 0.4*dataframe.shape[1]), 0.8*dataframe.shape[0] + 3))
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


def main(args):
    print("→ Iniciando pipeline de mantenimiento predictivo...")
    csv_path = args.csv
    outdir = args.outdir
    k = args.k

    ensure_dir(outdir)

    # 0) Cargar dataset
    df = pd.read_csv(csv_path)

    # Columna objetivo
    target_col = "fallo"
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el CSV. Columnas: {df.columns.tolist()}")

    # Columnas a descartar para modelado
    drop_cols = [c for c in ["timestamp", "machine_id"] if c in df.columns]

    # Columnas categóricas y numéricas
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c != target_col and c not in drop_cols]
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in [target_col] + drop_cols]

    # ===============================
    # 1) CLASIFICACIÓN FALLA 0/1
    # ===============================
    X = df.drop([target_col] + drop_cols, axis=1)
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    ohe_params = {"handle_unknown": "ignore"}
    if float(sklearn.__version__[:3]) < 1.4:
        ohe_params["sparse"] = False
    else:
        ohe_params["sparse_output"] = False

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**ohe_params), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
)

    clf = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=180,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])

    report_df.to_csv(os.path.join(outdir, "reporte_clasificador_fallo.csv"))
    cm_df.to_csv(os.path.join(outdir, "matriz_confusion_fallo.csv"))

    # Importancias globales del clasificador
    prep_fitted = clf.named_steps["prep"]
    rf_model = clf.named_steps["rf"]

    final_feature_names = []
    if categorical_cols:
        ohe = prep_fitted.named_transformers_["cat"]
        cat_out = ohe.get_feature_names_out(categorical_cols).tolist()
        final_feature_names.extend(cat_out)
    final_feature_names.extend(numeric_cols)

    fi = pd.Series(rf_model.feature_importances_, index=final_feature_names).sort_values(ascending=False)
    fi_df = fi.reset_index()
    fi_df.columns = ["feature", "importance"]
    fi_df.to_csv(os.path.join(outdir, "importancia_global_clasificador.csv"), index=False)

    plot_barh_series(
        fi.head(15),
        "Top 15 características más importantes (clasificador FALLA)",
        "Importancia",
        os.path.join(outdir, "top_importancias_clasificador_fallo.png"),
    )

    # ============================================
    # 2) CLUSTERING: SUBTIPOS de FALLA (solo num)
    # ============================================
    df_fallas = df[df[target_col] == 1].copy()
    num_for_clust = [c for c in numeric_cols if c in df_fallas.columns]
    if len(num_for_clust) == 0:
        raise ValueError("No hay columnas numéricas disponibles para clustering.")

    X_fallas_num = df_fallas[num_for_clust].copy()

    scaler = StandardScaler()
    Xf_scaled = scaler.fit_transform(X_fallas_num)

    # === Estimación automática de K ===
    possible_k = range(2, 101)  # probar entre 2 y 100 subtipos
    best_k, best_score = None, -1
    best_labels = None

    print("Buscando número óptimo de subtipos de falla...")
    for k in possible_k:
        mk = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, max_iter=100, n_init=5)
        labels = mk.fit_predict(Xf_scaled)
        try:
            score = silhouette_score(Xf_scaled, labels)
        except Exception:
            score = -1
        print(f"K={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"→ Mejor número de subtipos detectado: K={best_k} (silhouette={best_score:.3f})")
    df_fallas["subtipo_falla"] = best_labels

    # Export asignaciones de subtipo (unimos con columnas originales)
    asignacion_path = os.path.join(outdir, "asignacion_subtipos_falla.csv")
    df_fallas.to_csv(asignacion_path, index=False)

    # ===============================
    # 3) Sensores dominantes por subtipo
    # ===============================
    group_means = df_fallas.groupby("subtipo_falla")[num_for_clust].mean()
    global_means = df_fallas[num_for_clust].mean()
    global_stds = df_fallas[num_for_clust].std().replace(0, np.nan)

    z_diff = (group_means - global_means) / global_stds

    top_features_by_cluster = {}
    rows = []
    for cl in z_diff.index:
        ranked = z_diff.loc[cl].abs().sort_values(ascending=False)
        top_features_by_cluster[int(cl)] = ranked.head(10)
        for feat, val in ranked.head(10).items():
            rows.append({"subtipo_falla": int(cl), "feature": feat, "abs_z_diff": float(val)})

    dominance_df = pd.DataFrame(rows).sort_values(["subtipo_falla", "abs_z_diff"], ascending=[True, False])

    group_means.to_csv(os.path.join(outdir, "resumen_medias_por_subtipo.csv"))
    dominance_df.to_csv(os.path.join(outdir, "sensores_dominantes_por_subtipo.csv"), index=False)

    # Gráficos por subtipo
    plot_heatmap(group_means, f"Medias por subtipo de falla (K={k})", os.path.join(outdir, "heatmap_medias_por_subtipo.png"))
    for cl in sorted(top_features_by_cluster.keys()):
        ser = top_features_by_cluster[cl]
        plot_barh_series(
            ser,
            f"Subtipo {cl}: Top-10 sensores dominantes (|z-diff|)",
            "|z-diff| (desv. vs media global de FALLAS)",
            os.path.join(outdir, f"subtipo_{cl}_top10_sensores.png"),
        )

    # README resumen
    with open(os.path.join(outdir, "README_resultados_mantenimiento.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Pipeline ejecutado.\n\n"
            "1) Clasificador FALLA:\n"
            "- reporte_clasificador_fallo.csv, matriz_confusion_fallo.csv\n"
            "- importancia_global_clasificador.csv\n"
            "- top_importancias_clasificador_fallo.png\n\n"
            "2) Subtipos de FALLA:\n"
            "- asignacion_subtipos_falla.csv (registros con fallo=1 + subtipo)\n"
            "- resumen_medias_por_subtipo.csv\n"
            "- sensores_dominantes_por_subtipo.csv\n"
            "- heatmap_medias_por_subtipo.png\n"
            "- subtipo_<k>_top10_sensores.png\n"
        )

    print("✓ Pipeline finalizado correctamente")
    print(f"→ Resultados en: {os.path.abspath(outdir)}")
    print(f"→ Asignación de subtipos: {asignacion_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de mantenimiento predictivo (clasificación + clustering + explicabilidad)")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV con columna 'fallo'")
    parser.add_argument("--outdir", type=str, default="./salidas", help="Directorio de salida para artefactos")
    parser.add_argument("--k", type=int, default=3, help="Número de subtipos (clusters) para MiniBatchKMeans")
    args = parser.parse_args()
    main(args)
