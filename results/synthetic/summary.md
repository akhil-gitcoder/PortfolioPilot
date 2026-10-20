# Results (synthetic data)

Seeds: 30, evaluation budget: 10000. Hypervolume with pooled min-max normalisation, reference point 1.1.

| method        |   hv_mean |    hv_std |   hv_median |   igd_mean |   spacing_mean |   n_nd_mean |   runtime_mean_s |   max_violation |
|:--------------|----------:|----------:|------------:|-----------:|---------------:|------------:|-----------------:|----------------:|
| Hybrid        |   0.88516 |   0.00686 |     0.88383 |    0.03805 |        0.01978 |    251.967  |          9.88611 |               0 |
| NSGA2         |   0.53748 |   0.02777 |     0.54271 |    0.23021 |        0.01539 |     99.3667 |          1.73939 |               0 |
| NSGA2+current |   0.67087 |   0.01786 |     0.66912 |    0.1393  |        0.0189  |     99.5    |          1.87022 |               0 |
| NSGA2-tuned   |   0.8061  |   0.00951 |     0.8043  |    0.07128 |        0.02948 |     99.6667 |          1.92261 |               0 |
| RandomSearch  |   0.50519 |   0.01537 |     0.50277 |    0.23858 |        0.03023 |     63.7333 |          1.86741 |               0 |
| WeightedSum   |   0.88715 | nan       |     0.88715 |    0.05899 |        0.04971 |     66      |          1.47386 |               0 |


```json
{
  "data_source": "synthetic",
  "n_seeds": 30,
  "n_evals": 10000,
  "ws_hv": 0.8871502795504778,
  "seeds_RandomSearch_hv_greater_than_WS": 0,
  "seeds_NSGA2_hv_greater_than_WS": 0,
  "seeds_NSGA2+current_hv_greater_than_WS": 0,
  "seeds_NSGA2-tuned_hv_greater_than_WS": 0,
  "seeds_Hybrid_hv_greater_than_WS": 12,
  "MWU_NSGA2_vs_RandomSearch": {
    "U": 759.0,
    "p": 5.091173485363367e-06,
    "median_diff": 0.039940220069652255
  },
  "MWU_NSGA2+current_vs_NSGA2": {
    "U": 900.0,
    "p": 3.019859359162157e-11,
    "median_diff": 0.12640527894515674
  },
  "MWU_NSGA2-tuned_vs_NSGA2+current": {
    "U": 900.0,
    "p": 3.019859359162157e-11,
    "median_diff": 0.13518631566825612
  },
  "MWU_Hybrid_vs_NSGA2-tuned": {
    "U": 900.0,
    "p": 3.019859359162157e-11,
    "median_diff": 0.07952793967891703
  },
  "c_metric_median_run": {
    "RandomSearch": {
      "C(WS,method)": 0.9642857142857143,
      "C(method,WS)": 0.0
    },
    "NSGA2": {
      "C(WS,method)": 0.9393939393939394,
      "C(method,WS)": 0.0
    },
    "NSGA2+current": {
      "C(WS,method)": 0.8080808080808081,
      "C(method,WS)": 0.0
    },
    "NSGA2-tuned": {
      "C(WS,method)": 0.53,
      "C(method,WS)": 0.0
    },
    "Hybrid": {
      "C(WS,method)": 0.2190082644628099,
      "C(method,WS)": 0.0
    }
  }
}
```