# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9ea17f3aebef4af75f0a93c521d6e776070c7076063ed113dca780cff0b9684`
- fixture hash: `sha256-e43f09daa5c7f1f8012274d4f09baa27758aaa51c3e914baa4ee6b5329b895af`
- score hash: `sha256-75b1322238858670a75349fb6c0bb942c78e04ecacc5a33b4f357dca6ac0cbe3`
- bundle hash: `sha256-66215fc39548e317997bde4b1ab900a3afa50d5c42dfc969384ca5a36bb4934d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-174fe02cb9d576a687ddb560851b02ab0e12cb6737fa301408229ad552fa41d4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88972511e1f9411e57b15bac1d0d78b3ecb58ed9522521fd82c9d0ca15544549 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9e47ba9c10ecd33168616808d3592c84eeca1c871a995396ed7ef7f4768fa106 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-86d280713bb51211816204783080dd98df4ea40823364f9de13019fcc262d55a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-35aa6d4f | sha256-9dfa44dd53f824fb6585dc111a8d56f88e695e38358a42e205cacfb782b79aca |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-35aa6d4f | sha256-8ca6dc1f701775caf7ffa1e3e8a310aec70c5af52d3ce39517dcbbd538699e54 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c0831f74 | sha256-4e7c817bb7ad3a4d48d6b09231a596ed7331bc60b43aeff1e1fb0c53ee78277e |
