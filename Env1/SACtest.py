import EnvSAC
import numpy as np 
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device :",device)

agent = EnvSAC.SACAgent(7,2)
agent.load('SACparameters.pt')
env= EnvSAC.EnergyEnv()
state = env.reset()

T_log= list()
SOC_log = list()
grid_log= list()
price=0.0
done= False
total_reward=0.0

while not done:
    state_t = torch.tensor(state, dtype = torch.float32,device=device).unsqueeze(0)

    with torch.no_grad():
        action , _ = agent.actor.sample(state_t)
    action = action.squeeze(0).cpu().numpy()


    state, reward, done , _ = env.step(action)


    price += env.P_grid * EnvSAC.price_profile(max(0,env.hour-1))*env.dt

    total_reward+=reward
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

print(f"Total price with SAC agent: {price:.2f}")
print(f"Total reward for the episode : { total_reward:.2f}")