import pandas as pd
import glob
import json
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use("style.mplstyle")

input_files = glob.glob("data/**/**/results.json")

rows = []
for file in input_files:
    with open(file, "r") as f:
        data = json.load(f)
        rows.append(data)     
        
df = pd.DataFrame(rows)
print(df.to_string())

sns.relplot(
    data=df,
    x="num_qubits",
    y="time",
    hue="solver",
    kind="line",
)
plt.show()