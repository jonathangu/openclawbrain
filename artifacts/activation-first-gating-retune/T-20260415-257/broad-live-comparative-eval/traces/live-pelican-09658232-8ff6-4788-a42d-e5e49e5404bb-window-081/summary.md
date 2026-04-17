# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bed00a4e07440963fcb335df85735aa4cdf299e8eeeee26d6072510f9d967592`
- fixture hash: `sha256-7b9b3de84eea4b8f6489862313ae3b9c5b0de1ba49de793c86ffbf0e24eac4d6`
- score hash: `sha256-5ebcbf9e89005598ab6976db79f1748122a2dc6117762828086176fac6bc8933`
- bundle hash: `sha256-fd5dc397b76e8997bc1a668c3f58f3705035993cb959a90ba78c55d2704cffc1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7244838abdb700b7974c4fb03ae1270d3910510dcc5c175db289b9a82a5df872 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-19acd26b6455b31f96990d8a81e37ac018c608c9a4a21da65bc89b443c950a37 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d9b279a366b4c2300555be71255b0a72cd21582c00a29ac6bf1729498585a3e2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8262fa44f242eed9d5b0a96b5a49a2d5bee7caeded0c894abaafabb54d3ac98d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1f9618e2 | sha256-5932c4c91aaeec541b6122dd8075a21ddf633361d815f53d4fd9807c5ec47f21 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1f9618e2 | sha256-acb3a4266db0d0077eac255b168677cebe40d80b110c381b07f7988b42f47272 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-de2cf87f | sha256-fa850136145ba3d466eb1065f16d3a0b9a617ef8de51c104385e81f323bbd9e8 |
