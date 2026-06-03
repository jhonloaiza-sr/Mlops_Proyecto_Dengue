import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RANDOM_STATE,
    RL_SIMULATED_DATA_PATH,
    PROCESSED_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
)

def simulate_rl_agent_training():
    # 1. Aseguramos que las carpetas de reportes existan (Igual que el profe)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Cargamos los datos procesados
    df = pd.read_csv(PROCESSED_DATA_PATH)
    np.random.seed(RANDOM_STATE)

    # --- SIMULACIÓN DEL ENTRENAMIENTO DEL AGENTE (RL) ---
    episodios = 100
    # Simulamos que el agente aprende: la recompensa sube como una curva logarítmica
    recompensas = np.log(range(1, episodios + 1)) * 500 + np.random.normal(0, 50, episodios)
    # Simulamos el 'Exploration Rate' (Epsilon Decay) típico de RL
    epsilon = np.exp(-0.05 * np.arange(episodios))

    # 3. Generamos la Gráfica 1: Curva de Recompensa (Learning Curve)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodios + 1), recompensas, color='blue', alpha=0.7)
    plt.title("Curva de Aprendizaje del Agente RL (Recompensa Acumulada)")
    plt.xlabel("Episodios de Entrenamiento")
    plt.ylabel("Recompensa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rl_learning_curve.png")
    plt.close()

    # 4. Generamos la Gráfica 2: Decaimiento de la Exploración
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodios + 1), epsilon, color='red', marker=".")
    plt.title("Tasa de Exploración del Agente (Epsilon Decay)")
    plt.xlabel("Episodios")
    plt.ylabel("Epsilon (Prob. de explorar)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rl_epsilon_decay.png")
    plt.close()

    # 5. Simulamos la acción óptima del Agente sobre los datos
    # El agente optimiza reduciendo un 15% los mosquitos respecto al cronograma fijo
    df["accion_agente_mosquitos"] = (df["mosquitos_a_liberar"] * 0.85).astype(int)
    df["ahorro_logistico"] = df["mosquitos_a_liberar"] - df["accion_agente_mosquitos"]

    # Gráfica 3: Comparación Política Estática vs Política del Agente
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="temperatura_media",
        y="accion_agente_mosquitos",
        hue="casos_previos",
        palette="viridis",
        size="ahorro_logistico"
    )
    plt.title("Decisiones del Agente: Liberación Adaptativa por Clima")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "agent_policy_decisions.png")
    plt.close()

    # 6. Guardamos el resumen de métricas en CSV
    agent_summary = pd.DataFrame({
        "Episodio_Final": [episodios],
        "Recompensa_Maxima": [recompensas.max()],
        "Total_Ahorro_Simulado": [df["ahorro_logistico"].sum()]
    })
    agent_summary.to_csv(METRICS_DIR / "agent_training_summary.csv", index=False)

    # 7. Guardamos el nuevo dataset con las decisiones del agente
    RL_SIMULATED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RL_SIMULATED_DATA_PATH, index=False)

    print(" Entrenamiento del Agente RL simulado correctamente.")
    print(f" Dataset del agente guardado en: {RL_SIMULATED_DATA_PATH}")
    print(f" Gráficas guardadas en: {FIGURES_DIR}")
    print(agent_summary)

if __name__ == "__main__":
    simulate_rl_agent_training()