# app.py
from flask import Flask, render_template
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    if not os.path.exists("rewards.npy"):
        return "Primero ejecuta q_learning.py para entrenar el agente."

    rewards = np.load("rewards.npy")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Recompensa por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Entrenamiento del Agente Q-Learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/rewards_plot.png")
    plt.close()

    return render_template("resultado.html")

if __name__ == "__main__":
    app.run(debug=True)
