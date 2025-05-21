# q_learning.py
import numpy as np
import gym

def train_q_learning(episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    # Guardamos los resultados
    np.save("rewards.npy", rewards_per_episode)
    return Q, rewards_per_episode

if __name__ == "__main__":
    train_q_learning()
