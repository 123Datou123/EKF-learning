import numpy as np
from sir_model import sir_predict, jacobian_matrix
from data_utils import save_predictions

def predict_future_states(initial_state, initial_covariance, dt, Q, steps=7, output_file='future_predictions.csv'):
    x = initial_state.copy()
    P = initial_covariance.copy()
    predicted_states = []

    for _ in range(steps):
        x_pred = sir_predict(x, dt)
        F = jacobian_matrix(x, dt)
        P_pred = F @ P @ F.T + Q
        x = x_pred
        P = P_pred
        predicted_states.append(x.copy())

    predicted_states = np.array(predicted_states)
    save_predictions(predicted_states, output_file)

    return predicted_states

if __name__ == "__main__":
    # 初始状态和参数
    S0, I0, R0 = 0.99, 0.01, 0.0  # 初始状态
    beta0, gamma0 = 0.3, 0.1  # 初始参数
    dt = 0.1  # 时间步长

    # 初始状态向量和协方差矩阵
    initial_state = np.array([S0, I0, R0, beta0, gamma0])
    initial_covariance = np.eye(5) * 0.1
    initial_covariance[3, 3] = 0.5  # 初始传染率的较大不确定性
    initial_covariance[4, 4] = 0.5  # 初始恢复率的较大不确定性

    # 过程噪声协方差矩阵
    Q = np.eye(5) * 0.01
    Q[3, 3] = 0.05  # 传染率的不确定性
    Q[4, 4] = 0.05  # 恢复率的不确定性

    # 预测未来7个时间步长的状态和参数
    predicted_states = predict_future_states(initial_state, initial_covariance, dt, Q, steps=7)

    print("Predicted states for the next 7 steps:")
    for state in predicted_states:
        print(state)