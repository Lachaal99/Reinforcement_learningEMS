import EnvDQN
import numpy as np
import matplotlib.pyplot as plt
import torch
env= EnvDQN.EnergyEnv()
state= env.reset()
agent= EnvDQN.DQN(7,9)
agent.load("dqn_model.pth")
T_log= list()
SOC_log = list()
grid_log= list()
price=0.0

for t in range(24):
    with torch.no_grad():
        action = agent.greedy_action(state)

    state, reward, done , _ = env.step(action)

    price += env.P_grid * EnvDQN.price_profile(max(0,env.hour -1))
    T_log.append(env.T)
    SOC_log.append(env.SOC)
    grid_log.append(env.P_grid)

fig , ax = plt.subplots(3,1, figsize=(8,12))
ax[0].plot(T_log)
ax[0].axhline(21, linestyle="--")
ax[0].axhline(25, linestyle="--")
ax[0].set_ylabel("Indoor Temperature (°C)")
ax[0].set_xlabel("Hour")
ax[0].grid()


ax[1].plot(SOC_log)
ax[1].set_ylabel("Battery SOC")
ax[1].set_xlabel("Hour")
ax[1].grid()

ax[2].plot(grid_log)
ax[2].set_ylabel("Grid Power (kW)")
ax[2].set_xlabel("Hour")
ax[2].grid()

plt.show()

print(f"Total price with DQN agent: {price:.2f}")
