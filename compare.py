import pandas as pd 
import wandb
import numpy as np
import matplotlib.pyplot as plt
api = wandb.Api()
import argparse

"""
This script is used to compare the accuracy of the baseline and hierarchical models.

Usage:
    python compare.py -d <dataset> 

    or

    python compare.py -d <dataset> -nb <name of baseline run> -nh <name of hierarchical run>

"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, default="flowers", help="Dataset to compare")
parser.add_argument("--name-base", "-nb", type=str, default="none", help="Name of the baseline run")
parser.add_argument("--name-hier", "-nh", type=str, default="none", help="Name of the hierarchical run")

args = parser.parse_args()
dataset = args.dataset


runs = api.runs("procom/procom-transformers")
for run in runs:

    if run.config.get("dataset") is None:
        continue

    if run.config["dataset"] == dataset:
        if run.config["type"] == "hierarchical" and (run.name == args.name_hier or args.name_hier == "none"):
            name_hier = run.id
        elif run.config["type"] == "baseline" and (run.name == args.name_base or args.name_base == "none"):
            name_base = run.id
        else:
            print("Wrong type")


print(f"Comparing {name_base} and {name_hier}")

run_b = api.run(f"/procom/procom-transformers/runs/{name_base}")

history_base = list(run_b.scan_history(keys=["running_accuracy"]))
data_base = pd.DataFrame.from_records(history_base).to_numpy()

run_h = api.run(f"/procom/procom-transformers/runs/{name_hier}")

history_hier = list(run_h.scan_history(keys=["running_accuracy"]))
data_hier = pd.DataFrame.from_records(history_hier).to_numpy()

mean_base = np.mean(data_base)
mean_hier = np.mean(data_hier)

t = np.linspace(min(np.min(data_base), np.min(data_hier)), max(np.max(data_base), np.max(data_hier)), 1000)

plt.plot(t,t, 'k--')
plt.scatter(data_base[data_base<data_hier], data_hier[data_base<data_hier], c='r', label='H > B')
plt.scatter(data_base[data_base>data_hier], data_hier[data_base>data_hier], c='b', label='B > H')
plt.scatter(data_base[data_base==data_hier], data_hier[data_base==data_hier], c='g', label='B = H')
plt.xlabel('Base')
plt.ylabel('Hierarchical')
plt.title(f"Accuracy comparison for {dataset} dataset (b/h:{mean_base:.2f}/{mean_hier:.2f})")
plt.legend()
plt.savefig(f"temp/compare_{dataset}.png", dpi=300)

