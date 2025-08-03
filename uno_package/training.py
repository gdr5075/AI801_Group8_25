import os
from datetime import datetime
import json

def train_multiple_iterations(envToTrain, iterations):
    results_log = []
    for i in range(iterations):
        result = envToTrain.train()
        # Log key metrics
        print(f"Iteration {i+1}")
        print("Episode reward mean:", result["env_runners"]["episode_return_mean"])
        print("Episode length mean:", result["env_runners"]["episode_len_mean"])
        print("num_env_steps_sampled:", result["env_runners"]["num_env_steps_sampled"])
        print("---")
        # Save results for later analysis
        timestamp =datetime.now().timestamp()
        results_log.append({
            "iteration": i+1,
            "timestamp": timestamp,
            "episode_return_mean": result["env_runners"]["episode_return_mean"],
            "episode_len_mean": result["env_runners"]["episode_len_mean"],
            "num_env_steps_sampled": result["env_runners"]["num_env_steps_sampled"],
        })
        # Optionally, save results_log to a file for later plotting
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #Save
        checkpoint_path = envToTrain.save_to_path(f"file://{dir_path}/../checkpoints/checkpoint_{timestamp}")
        print("checkpoint saved at", checkpoint_path)

    with open(f'{dir_path}/../checkpoints/checkpoint_{timestamp}/training_results_{timestamp}.json', "w") as f:
        json.dump(results_log, f, indent=2)