from SACAgent import SACAgent
import Env
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device :",device)

env = Env.EnergyEnv()
state = env.reset()
agent = SACAgent(9,3)
episode_num = 5000
episode_rewards= []
episode_prices = []


for episode in range(episode_num):

    total_reward = 0.0
    total_price=0.0
    state= env.reset()
    done= False
    while not done:
        if len(agent.buffer)<10000:
            action = env.sample_random_action()
        else:
            state_t = torch.tensor(state, dtype= torch.float32,device=device).unsqueeze(0)
            with torch.no_grad():
                action , _ = agent.actor.sample(state_t)
                action = action.squeeze(0).cpu().numpy()
        next_state, reward, done, _ =env.step(action)



        agent.buffer.push(state, action , reward, next_state,done)
        if len(agent.buffer) > 10000:
            agent.update()


        state = next_state
        total_reward += reward
        total_price += env.P_grid * Env.price_profile(max(0,env.hour -1))*env.dt

        # append rewards
    episode_rewards.append(total_reward)
    episode_prices.append(total_price)
    if episode %5 ==0 : 
        print(f" Episode {episode},Total Reward : {total_reward:.2f}, total Price : {total_price:.2f}")

agent.save()



# plot rewards
window = 50
moving_avg_rewards = np.convolve(
    episode_rewards, 
    np.ones(window)/window, 
    mode="valid"
    )
moving_avg_prices = np.convolve(
    episode_prices,
    np.ones(window)/window,
    mode="valid"
    )



figure , ax = plt.subplots(2,1, figsize=(8,10))
ax[0].plot(episode_rewards, alpha=0.3, label="Episode reward")
ax[0].plot(range(window-1, len(episode_rewards)), moving_avg_rewards, label="Moving avg")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Total reward")
ax[0].legend()
ax[0].grid()
ax[1].plot(episode_prices, alpha=0.3, label="Episode price")
ax[1].plot(range(window-1, len(episode_prices)), moving_avg_prices, label="Moving avg")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Total price")
ax[1].legend()
ax[1].grid()
plt.show()