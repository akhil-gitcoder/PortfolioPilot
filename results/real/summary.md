# Results (real data)

Seeds: 30, evaluation budget: 10000. Hypervolume with pooled min-max normalisation, reference point 1.1.

| method        |   hv_mean |    hv_std |   hv_median |   igd_mean |   spacing_mean |   n_nd_mean |   runtime_mean_s |   max_violation |
|:--------------|----------:|----------:|------------:|-----------:|---------------:|------------:|-----------------:|----------------:|
| Hybrid        |   0.88137 |   0.00584 |     0.88279 |    0.03733 |        0.02037 |    269.933  |          2.28388 |               0 |
| NSGA2         |   0.53238 |   0.0271  |     0.52618 |    0.22818 |        0.02056 |     99.5333 |          0.35302 |               0 |
| NSGA2+current |   0.67393 |   0.02081 |     0.67245 |    0.15199 |        0.0199  |     99.4667 |          0.36495 |               0 |
| NSGA2-tuned   |   0.80073 |   0.01048 |     0.80134 |    0.06852 |        0.02989 |     99.8667 |          0.42848 |               0 |
| RandomSearch  |   0.51621 |   0.00926 |     0.51785 |    0.23068 |        0.02879 |     85.4333 |          0.36111 |               0 |
| WeightedSum   |   0.87705 | nan       |     0.87705 |    0.06724 |        0.05223 |     62      |          0.61798 |               0 |


```json
{
  "data_source": "real",
  "n_seeds": 30,
  "n_evals": 10000,
  "ws_hv": 0.8770516535062132,
  "seeds_RandomSearch_hv_greater_than_WS": 0,
  "seeds_NSGA2_hv_greater_than_WS": 0,
  "seeds_NSGA2+current_hv_greater_than_WS": 0,
  "seeds_NSGA2-tuned_hv_greater_than_WS": 0,
  "seeds_Hybrid_hv_greater_than_WS": 21,
  "MWU_NSGA2_vs_RandomSearch": {
    "U": 617.0,
    "p": 0.013831619802975413,
    "median_diff": 0.008324758201785132
  },
  "MWU_NSGA2+current_vs_NSGA2": {
    "U": 900.0,
    "p": 3.0198593591621564e-11,
    "median_diff": 0.14627732521085557
  },
  "MWU_NSGA2-tuned_vs_NSGA2+current": {
    "U": 900.0,
    "p": 3.0198593591621564e-11,
    "median_diff": 0.12888880706965355
  },
  "MWU_Hybrid_vs_NSGA2-tuned": {
    "U": 900.0,
    "p": 3.0198593591621564e-11,
    "median_diff": 0.08145072344174287
  },
  "c_metric_median_run": {
    "RandomSearch": {
      "C(WS,method)": 0.8941176470588236,
      "C(method,WS)": 0.0
    },
    "NSGA2": {
      "C(WS,method)": 0.82,
      "C(method,WS)": 0.0
    },
    "NSGA2+current": {
      "C(WS,method)": 0.7373737373737373,
      "C(method,WS)": 0.0
    },
    "NSGA2-tuned": {
      "C(WS,method)": 0.6,
      "C(method,WS)": 0.0
    },
    "Hybrid": {
      "C(WS,method)": 0.22053231939163498,
      "C(method,WS)": 0.0
    }
  }
}
```