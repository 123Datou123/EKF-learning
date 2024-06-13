import numpy as np
import csv

def generate_sir_data(days, S0, I0, R0, beta, gamma, dt):
    S = S0
    I = I0
    R = R0
    data = []

    for day in range(days):
        S_next = S - beta * S * I * dt
        I_next = I + (beta * S * I - gamma * I) * dt
        R_next = R + gamma * I * dt

        data.append([S_next, I_next, R_next])

        S = S_next
        I = I_next
        R = R_next

    return np.array(data)

# 设置参数
days = 200
S0 = 0.99
I0 = 0.01
R0 = 0.0
beta = 0.3
gamma = 0.1
dt = 1.0

# 生成数据
data = generate_sir_data(days, S0, I0, R0, beta, gamma, dt)

# 保存数据到CSV文件
with open('observations.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["S", "I", "R"])  # 写入表头
    for row in data:
        writer.writerow(row)

print("Generated test data for at least 200 days and saved to observations.csv")
