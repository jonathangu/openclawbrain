# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3f86fd026217c7d6458e87e96268ca58f7633ecf498ef1f8793a6a7617c13f8`
- fixture hash: `sha256-c25bf3a6bec00b35ab13366d1787d21cc5e0fb28011aa90689176fbd43238498`
- score hash: `sha256-cc5d9233badfce980207834ed865a3355736085af744e00cb4583b864c58b0f7`
- bundle hash: `sha256-192de5ae6c5c2c2ff30d40ce7d4c4bb86e56d23045d340b4b6ba0632b18a6eb2`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e36e95d3b902dbb1cba84b7196a751790c689dc2e631e7340724bc6d85c3a59 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd458346792fbe39b57e5f1016980c1af8ed4e2c44d270bc21c75bbaee76f5e8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-de7b38d08def425e0c8ff6b34db3fcce8d0740c3351da4db87ea8ba6cfb08e4c |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-84aa1593c58e3f630275d5d19af9a0ae5945a8542a4c8f42d1ad0f2448b77eca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a36c2361 | sha256-618f14ce6f01d4d954cc7f92f9b24a0f452ac51079e9b1571d41f0bb68f455c5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a36c2361 | sha256-d8f9e5946972f7007c56b62a929a49a8234ef18d19058de1684019f30da641dc |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-201c4706 | sha256-60d615ae1bdeced7b74241e51baff042b15af2c6c382a9715c85261f9470b553 |
