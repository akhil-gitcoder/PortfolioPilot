# Weekly progress log

| Week | Dates | Work done | Members | Evidence |
|---|---|---|---|---|
| 1 | 19 to 25 Oct | Topic, formulation, project structure, data layer, problem model with repair, NSGA-II from scratch, random-search and weighted-sum baselines, metrics, 28 tests, first synthetic-data results | Akhil | git history up to c. 22 Oct |
| 2 | 26 Oct to 1 Nov | Mid-evaluation preparation, commit walkthrough, explanation notes | Akhil, Ridhima | mid-eval notes |
| 3 | 2 to 8 Nov | Real-data download and DATA.md, cost rates, 30-seed real experiment, NSGA-II improvements (inject current, sparse init, reset mutation), external archive, hybrid polish, memory fix in pooling, walk-forward test | Ridhima | commits through c73eddb |
| 4 | 9 Nov onwards | Final report, README, formulation and references, presentation, fresh-clone test | Akhil, Ridhima | to be filled |

## Findings that shaped the work
- Plain NSGA-II lost to the weighted-sum baseline; diagnosis: random starts held most of the 30 stocks. Fixed step by step; only the hybrid with archive beat the baseline.
- A cost-multiplier sensitivity study was dropped: scaling an objective by a constant cannot change the Pareto front.
- Walk-forward showed no out-of-sample advantage; reported as is.
- Lamarckian write-back, seeding and a larger budget were tested and not adopted.
