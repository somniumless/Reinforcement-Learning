from flask import render_template

@app.route("/results")
def show_results():
    # Simulación de datos (reemplaza con tus datos reales)
    rewards = [10, 20, 15, 30, 25]  # Ejemplo: recompensas por episodio
    portfolio_values = [10000, 10500, 11000, 11500, 12000]  # Ejemplo: evolución del balance
    
    return render_template(
        "resultado.html",
        rewards=rewards,
        portfolio_values=portfolio_values
    )

@app.route("/train", methods=["POST"])
def train():
    episodes = int(request.form.get("episodes", 100))
    rewards = []
    portfolio_values = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            portfolio_values.append(env.balance + (env.shares_held * env.current_price))  # Guarda el valor del portafolio

        rewards.append(episode_reward)

    return jsonify({
        "status": "success",
        "rewards": rewards,
        "portfolio_values": portfolio_values
    })