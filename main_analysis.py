import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import axes

# Create the figures directory
os.makedirs("figures", exist_ok=True)

input_files = glob.glob("data/**/results.json", recursive=True)

# Create a DataFrame from the JSON files
rows = []
for file in input_files:
    with open(file, "r") as f:
        data = json.load(f)
        rows.append(data)

df = pd.DataFrame(rows)

# Figure 1: Number of Qubits vs Time for different solvers
sns.relplot(
    data=df[df["circuit-type"] == "brickwork"],
    x="num_qubits",
    y="time",
    hue="solver",
    kind="line",
).set(title="num_layers=10")

plt.savefig("figures/num_qubits_time_solvers.png", dpi=300)
plt.show()

# Figure 2: Number of Qubits vs tiem for different circuit types
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

filtered_df = df[(df["solver"] == "cwmc") & (df["num_qubits"] <= 8)]

sns.lineplot(
    ax=axes[0],
    data=filtered_df,
    x="num_qubits",
    y="time",
    hue="circuit-type",
)
axes[0].set_title("num_layers=10")

sns.lineplot(
    ax=axes[1],
    data=filtered_df,
    x="num_qubits",
    y="cnf_num_vars",
    hue="circuit-type",
)
axes[1].set_title("num_layers=10")

sns.lineplot(
    ax=axes[2],
    data=filtered_df,
    x="num_qubits",
    y="cnf_num_clauses",
    hue="circuit-type",
)
axes[2].set_title("num_layers=10")

plt.tight_layout()
plt.savefig("figures/preliminary_results.png", dpi=300)
plt.show()
