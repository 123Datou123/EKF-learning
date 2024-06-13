import numpy as np

def sir_predict(x,dt):
    S, I, R, beta, gamma = x
    S_next = S - beta * S * I * dt
    I_next = I + (beta * S * I - gamma * I) * dt
    R_next = R + gamma * I * dt
    return np.array([S_next, I_next, R_next, beta, gamma])

def jacobian_matrix(x, dt):
    S, I, R, beta, gamma = x
    F = np.array([
        [1 - beta * I * dt, -beta * S * dt, 0, -S * I * dt, 0],
        [beta * I * dt, 1 + (beta * S - gamma) * dt, 0, S * I * dt, -I * dt],
        [0, gamma * dt, 1, 0, I * dt],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    return F