""" 
Plotting functions for analyzing quantum circuit performance.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Create the figures directory
os.makedirs("figures", exist_ok=True)

input_files = glob.glob("data-figures/**/results.json", recursive=True)

# Create a DataFrame from the JSON files
rows = []
for file in input_files:
    with open(file, "r") as f:
        data = json.load(f)
        rows.append(data)

df = pd.DataFrame(rows)

# Create a combined label for hue
df['solver_label'] = df.apply(
    lambda row: f"{row['solver']} ({row['encoding-method']})" if row['solver'] == 'cwmc' else row['solver'],
    axis=1
)

# Figure 1

g = sns.relplot(
    data=df[df["circuit-type"] == "semi-brickwork"],
    x="num-qubits",
    y="time",
    hue="brickwork-ratio",
    kind="line",
    markers=True,
    errorbar="se",
    style="brickwork-ratio"
).set(yscale="log", xscale="log")
plt.grid()
g.legend.set_title("r")
g.set_axis_labels("Number of qubits", "Time (s)")
print(df[df["circuit-type"] == "semi-brickwork"].columns)
plt.savefig("figures/brickwork-ratio.pdf")
plt.show()


# Figure 2

palette = sns.color_palette()
g = sns.relplot(
    data=df[df["circuit-type"] == "cnot-brickwork"],
    x="num-qubits",
    y="time",
    hue="solver_label",
    hue_order=["cwmc (valid-paths)", "sv", "tn"],
    kind="line",
    markers=True,
    errorbar="se",
    style="solver_label",
    palette=[palette[1], palette[2], palette[3]]
).set(yscale="log", xscale="log") 
plt.grid()
g.set_axis_labels("Number of qubits", "Time (s)")
g.legend.set_title("Method")
g.legend.texts[0].set_text("WMC - Valid paths")
g.legend.texts[1].set_text("Statevector")
g.legend.texts[2].set_text("Tensor networks")
plt.savefig("figures/cnot-brickwork.pdf")
plt.show()

# Figure 3

g = sns.relplot(
    data=df[(df["circuit-type"] == "brickwork") & (df["num-qubits"] == df["num-layers"])],
    x="num-qubits",
    y="time",
    hue="solver_label",
    hue_order=["cwmc (all-paths)", "cwmc (valid-paths)", "sv", "tn"],
    kind="line",
    style="solver_label",
    markers=True,
    errorbar="se",
)
g.set_axis_labels("Number of qubits", "Time (s)")
g.legend.set_title("Method")
g.legend.texts[0].set_text("WMC - All paths")
g.legend.texts[1].set_text("WMC - Valid paths")
g.legend.texts[2].set_text("Statevector")
g.legend.texts[3].set_text("Tensor networks")
g.ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 4]))
plt.grid()
plt.savefig("figures/solvers.pdf", bbox_inches='tight')
plt.show()
