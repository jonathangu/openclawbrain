# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6cd785628fb3c34642dd7b4a701799a6e96acb06e347a7bf1d01cd4950a8de4c`
- fixture hash: `sha256-8169ace4aebbc5a4a546b5c0d2bdc7c5a395f1f1630a066be79c7f63594673d2`
- score hash: `sha256-35a8988b6e57eae30a452fe7250b4c5ed85b62290dde47084a825dda33ecb1c5`
- bundle hash: `sha256-695d2154b5b190dd1c102a6816b4af8b216e9816eb0fae786414b447afeaaede`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6fbe7613b07ee3e659c5ab0ce2fe9e83640dc0dbe17b255f0c268784354a36 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-5a3e56b3d1a60ccbf8b8dd8cc1bba24e51512f6c6eb432df4e1946fd204d9cee |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-efe8a3efcc79b19d73f784535b597425c9ed034a27f039aea0b3996ad37a60c7 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-76fb6571de650c513abc73b164ec434e8d6eaaa3e6356d1deee16848573bc297 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-34c80b1b | sha256-29f425e3865f88f42f43874bddeb10fe45e9c1794d30f60101212c9349a80aa7 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-34c80b1b | sha256-b58e6cb654ef80e0248e10facd66e2084224a75c29e9df6598a7e2de3f2d11d0 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-eb5024e2 | sha256-4f2183ade66e8f2952208bdfbde08de145d1e6d171da11ccf47aaf474d920ca0 |
