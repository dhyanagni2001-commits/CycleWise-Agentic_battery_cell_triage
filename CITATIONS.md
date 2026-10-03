# Citations

This is the closed list of sources agents may cite. The citation validator
(`cyclewise/agents/citations.py`) parses the `id` column and rejects any
citation key that is not in this table. DOIs were checked against doi.org
metadata on 2026-10-03.

| id | Reference | DOI / URL | Used for |
|---|---|---|---|
| severson2019 | K. A. Severson, P. M. Attia, N. Jin, N. Perkins, et al. "Data-driven prediction of battery cycle life before capacity degradation." *Nature Energy* 4, 383–391 (2019). | https://doi.org/10.1038/s41560-019-0356-8 | Dataset; ΔQ(V) variance feature; 80% end-of-life definition; baseline model |
| attia2020 | P. M. Attia, A. Grover, N. Jin, K. A. Severson, et al. "Closed-loop optimization of fast-charging protocols for batteries with machine learning." *Nature* 578, 397–402 (2020). | https://doi.org/10.1038/s41586-020-1994-5 | Early prediction used to stop tests early; closed-loop budgeted testing |
| severson2019_data | Toyota Research Institute, "Data-driven prediction of battery cycle life before capacity degradation" dataset, data.matr.io (CC BY 4.0). | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 | Raw data files for batches 2017-05-12 and 2017-06-30 |
| severson2019_code | Authors' reference loader and feature code. | https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation | Continued-cell merge, censored-cell list, noisy-channel exclusions |
| hacknation_c03 | Hack-Nation x Databricks, Challenge 03 "Agentic Scientific Discovery" brief. | (challenge brief, provided to participants) | Problem framing, judging criteria |
| omnigent | Omnigent documentation (open-source meta-harness, Databricks). | https://omnigent.ai/docs | Agent, tool, and policy definitions |
