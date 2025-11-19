import { useEffect, useState } from "react";

export default function App() {
  const [data, setData] = useState(null);

  // Llama al backend cada 1 segundo
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/stream");
        const json = await res.json();
        if (json.status !== "esperando_datos") setData(json);
      } catch (err) {
        console.error("Error al consultar backend:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (!data)
    return (
      <div style={{ padding: 20 }}>
        <h2>Esperando lecturas...</h2>
      </div>
    );

  const lectura = data.lectura_original;
  const resultado = data.resultado;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Monitoreo de Sensores</h1>

      <section style={styles.card}>
        <h2 style={styles.subtitle}>Probabilidad de Falla</h2>

        {resultado.es_falla ? (
          <>
            <p><b>Tipo de Falla Detectada:</b> {resultado.tipo_falla}</p>
            <p><b>Subtipo:</b> {resultado.subtipo_id}</p>
            <div style={styles.progressContainer}>
              <div style={{ ...styles.progressBar, width: `${resultado.similitud_pct}%`,
                    backgroundColor: resultado.similitud_pct > 70 ? "#e74c3c" :
                                     resultado.similitud_pct > 40 ? "#f1c40f" : "#2ecc71" }}>
              </div>
            </div>
            <p><b>Similitud:</b> {resultado.similitud_pct.toFixed(2)}%</p>
          </>
        ) : (
          <p style={{ color: "#2ecc71", fontWeight: "bold" }}>Operación normal ✓</p>
        )}
      </section>

      <section style={styles.card}>
        <h2 style={styles.subtitle}>Sensores</h2>

        {Object.entries(lectura).map(([key, value]) => (
          <div key={key} style={styles.sensorRow}>
            <span style={styles.sensorName}>{key}:</span>
            <span style={styles.sensorValue}>{String(value)}</span>
          </div>
        ))}
      </section>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "system-ui, sans-serif",
    padding: 20,
    maxWidth: 800,
    margin: "0 auto"
  },
  title: {
    textAlign: "center",
    marginBottom: 20
  },
  card: {
    background: "#fff",
    padding: 20,
    borderRadius: 10,
    marginBottom: 20,
    boxShadow: "0 3px 10px rgba(0,0,0,0.1)"
  },
  subtitle: {
    marginBottom: 10,
    borderBottom: "solid 1px #ddd",
    paddingBottom: 6
  },
  sensorRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
    borderBottom: "solid 1px #eee"
  },
  sensorName: {
    fontWeight: "bold"
  },
  sensorValue: {
    color: "#555"
  },
  progressContainer: {
    width: "100%",
    height: 18,
    background: "#eee",
    borderRadius: 6,
    marginTop: 10,
    marginBottom: 10,
    overflow: "hidden"
  },
  progressBar: {
    height: "100%",
    transition: "0.4s ease"
  }
};
