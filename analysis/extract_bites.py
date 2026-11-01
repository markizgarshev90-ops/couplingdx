"""Extract bite start times from the derived FIC timeline workbook."""
import os
import numpy as np, pandas as pd, openpyxl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wb = openpyxl.load_workbook(os.path.join(ROOT, "data", "fic_bite_timelines.xlsx"), read_only=True)
rows, meals = [], []
for i in range(21):
    ws = wb[f"entry_{i}"]
    t, c = [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        t.append(r[0]); c.append(r[1])
    t = np.asarray(t, float); c = np.asarray(c, float)
    d = np.diff(c)
    idx = np.where(d > 0)[0] + 1
    reps = d[idx - 1].astype(int)
    times = np.repeat(t[idx], reps)
    rows += [dict(meal=i, t=x) for x in times]
    meals.append(dict(meal=i, T=t[-1], N=len(times)))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data", "fic_bite_times.csv"), index=False)
pd.DataFrame(meals).to_csv(os.path.join(ROOT, "data", "fic_meals.csv"), index=False)
print(pd.DataFrame(meals).describe())
