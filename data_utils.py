import numpy as np

def load_observations(file_path):
    return np.loadtxt(file_path, delimiter=',')

def save_predictions(predictions, file_path):
    np.savetxt(file_path, predictions, delimiter=',')