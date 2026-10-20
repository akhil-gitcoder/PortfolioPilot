# Results (synthetic data)

Seeds: 4, evaluation budget: 3000. Hypervolume with pooled min-max normalisation, reference point 1.1.

| method               |   hv_mean |    hv_std |   hv_median |   igd_mean |   spacing_mean |   n_nd_mean |   runtime_mean_s |   max_violation |
|:---------------------|----------:|----------:|------------:|-----------:|---------------:|------------:|-----------------:|----------------:|
| NSGA2                |   0.37872 |   0.01231 |     0.38004 |    0.38273 |        0.01129 |        98.5 |          0.45336 |               0 |
| NSGA2+current        |   0.48707 |   0.01677 |     0.4824  |    0.09253 |        0.01623 |       100   |          0.47048 |               0 |
| NSGA2+current+polish |   0.62666 |   0.01242 |     0.62362 |    0.05281 |        0.01303 |       218   |          8.13104 |               0 |
| RandomSearch         |   0.41717 |   0.01691 |     0.41863 |    0.3441  |        0.0347  |        42.5 |          0.49318 |               0 |
| WeightedSum          |   0.79691 | nan       |     0.79691 |    0.05099 |        0.05915 |        66   |          1.47051 |               0 |


```json
{
  "data_source": "synthetic",
  "n_seeds": 4,
  "n_evals": 3000,
  "ws_hv": 0.7969068457484897,
  "seeds_RandomSearch_hv_greater_than_WS": 0,
  "seeds_NSGA2_hv_greater_than_WS": 0,
  "seeds_NSGA2+current_hv_greater_than_WS": 0,
  "seeds_NSGA2+current+polish_hv_greater_than_WS": 0,
  "MWU_NSGA2_vs_RandomSearch": {
    "U": 0.0,
    "p": 0.02857142857142857,
    "median_diff": -0.03858946337558644
  },
  "MWU_NSGA2+current_vs_NSGA2": {
    "U": 16.0,
    "p": 0.02857142857142857,
    "median_diff": 0.10236388764944515
  },
  "MWU_NSGA2+current+polish_vs_NSGA2+current": {
    "U": 16.0,
    "p": 0.02857142857142857,
    "median_diff": 0.14121488533163518
  },
  "c_metric_median_run": {
    "RandomSearch": {
      "C(WS,method)": 1.0,
      "C(method,WS)": 0.0
    },
    "NSGA2": {
      "C(WS,method)": 1.0,
      "C(method,WS)": 0.0
    },
    "NSGA2+current": {
      "C(WS,method)": 0.76,
      "C(method,WS)": 0.0
    },
    "NSGA2+current+polish": {
      "C(WS,method)": 0.4752475247524752,
      "C(method,WS)": 0.0
    }
  }
}
```