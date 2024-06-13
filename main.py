import numpy as np
from sir_model import sir_predict, jacobian_matrix
from data_utils import load_observations, save_predictions
from prediction import predict_future_states
from visualize import load_predictions

# 初始状态和参数
S0, I0, R0 = 0.99, 0.01, 0.0  # 初始状态
beta0, gamma0 = 0.15, 0.1  # 初始参数
dt = 1.0  # 时间步长

# 初始状态向量和协方差矩阵
x = np.array([S0, I0, R0, beta0, gamma0])
P = np.eye(5) * 0.1
P[3, 3] = 0.25  # 初始传染率的较大不确定性
P[4, 4] = 0.25  # 初始恢复率的较大不确定性

# 过程噪声协方差矩阵和观测噪声协方差矩阵
Q = np.eye(5) * 0.01
Q[3, 3] = 0.05  # 传染率的不确定性
Q[4, 4] = 0.05  # 恢复率的不确定性
R = np.eye(3) * 0.01  # 对应 S, I, R 的观测噪声

# 加载观测数据
observations = load_observations('observations.csv')

# 用于记录每一步的状态向量
recorded_states = []

# 执行卡尔曼滤波
for z in observations[:40]:
    # 预测步骤
    x_pred = sir_predict(x, dt)
    F = jacobian_matrix(x, dt)
    P_pred = F @ P @ F.T + Q

    # 更新步骤
    H = np.array([
        [1, 0, 0, 0, 0],  # 对应 S 的观测矩阵
        [0, 1, 0, 0, 0],  # 对应 I 的观测矩阵
        [0, 0, 1, 0, 0]   # 对应 R 的观测矩阵
    ])
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x = x_pred + K @ (z - H @ x_pred)
    P = (np.eye(5) - K @ H) @ P_pred

    # 记录更新后的状态
    recorded_states.append(x.copy())

    # 输出更新后的状态
    print(f"Updated state: {x}")

# 保存预测结果
save_predictions(np.array(recorded_states), 'predicted_states.csv')

# 调用预测函数进行未来7个时间步长的预测，并在文件 'future_predictions.csv' 中保存结果
predict_future_states(x, P, dt, Q, steps=7)
#
# future_predictions = load_predictions('future_predictions.csv')
# print("Predicted states for the next 7 days:")
# for day, state in enumerate(future_predictions, start=1):
#     print(f"Day {day}: {state}")
