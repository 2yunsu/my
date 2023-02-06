import numpy as np
import pandas as pd
import math
from scipy.stats import norm, cauchy
from tqdm import tqdm

graph = []

for i in tqdm(range(1000)):
    #초기값
    n = 100
    a = 0.05
    d = cauchy.rvs(size = n)
    eps = math.sqrt((1/(2*n)) * math.log(2/a))

    Fn = lambda x : sum(d < x) / n
    F_max = lambda x : max(Fn(x) - eps, 0)
    F_min = lambda x : min(Fn(x) + eps, 1)

    sort = sorted(d)

    df = pd.DataFrame({
        'x': sort,
        'F': np.array(list(map(Fn, sort))),
        'F_min': np.array(list(map(F_min, sort))),
        'F_max': np.array(list(map(F_max, sort))),
        'CDF': np.array(list(map(norm.cdf, sort)))
    })
    data = ((df['F_min'] >= df['CDF']) & (df['CDF'] >= df['F_max'])).all()
    graph.append(data)

    Average = np.array(graph).mean()
print("Cauchy_Average:", Average)



