# visualize.py

import numpy as np
import matplotlib.pyplot as plt

def load_predictions(file_path):
    return np.loadtxt(file_path, delimiter=',')

def plot_states(recorded_states, title):
    time_steps = range(len(recorded_states))
    S = recorded_states[:, 0]
    I = recorded_states[:, 1]
    R = recorded_states[:, 2]
    beta = recorded_states[:, 3]
    gamma = recorded_states[:, 4]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, S, label='Susceptible (S)')
    plt.plot(time_steps, I, label='Infected (I)')
    plt.plot(time_steps, R, label='Recovered (R)')
    plt.xlabel('Time Steps')
    plt.ylabel('Population Proportion')
    plt.legend()
    plt.title(f'{title} - SIR Model')

    plt.subplot(2, 1, 2)
    plt.plot(time_steps, beta, label='Transmission Rate (beta)')
    plt.plot(time_steps, gamma, label='Recovery Rate (gamma)')
    plt.xlabel('Time Steps')
    plt.ylabel('Rates')
    plt.legend()
    plt.title(f'{title} - Transmission and Recovery Rates')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    recorded_states = load_predictions('predicted_states.csv')
    plot_states(recorded_states, 'Predicted States')

    future_predictions = load_predictions('future_predictions.csv')
    plot_states(future_predictions, 'Future Predictions')
