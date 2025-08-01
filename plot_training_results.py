import json
import matplotlib.pyplot as plt

# Load training results from file
with open("training_results.json", "r") as f:
    results = json.load(f)

iterations = [entry["iteration"] for entry in results]
episode_reward_mean = [entry["episode_return_mean"] for entry in results]
episode_len_mean = [entry["episode_len_mean"] for entry in results]
timesteps_total = [entry["timesteps_total"] for entry in results]

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(iterations, episode_reward_mean, marker='o')
plt.title("Episode Reward Mean per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Episode Reward Mean")

plt.subplot(2, 1, 2)
plt.plot(iterations, episode_len_mean, marker='o', color='orange')
plt.title("Episode Length Mean per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Episode Length Mean")

plt.tight_layout()
plt.savefig("training_results_plot.png")
print("Plot saved as training_results_plot.png")
