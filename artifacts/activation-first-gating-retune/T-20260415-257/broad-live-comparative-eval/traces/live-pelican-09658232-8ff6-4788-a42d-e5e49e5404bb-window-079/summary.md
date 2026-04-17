# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-81d89529d4ba3551ffef2373c3a90591f4a3287648e2c06c75e207e29f8e1526`
- fixture hash: `sha256-ef31f66fdeb7a284c6c5e031c684ec09c55fa37e67e6013d84cbbb7caa013474`
- score hash: `sha256-91166a05bef175c0bfcc80a767688a6aadf37b9a8ab0723d58db767c5c4d3896`
- bundle hash: `sha256-eb51de16158fa19ad6062dae3be0a6b7e0ab1ebdc64d546b0785ed880ade3275`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b4b5bf35e2ebcc8ce20efe2342fbd2d24f5f0b713e668d8e9bc9cfb1b1256e40 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-956909b87b854263a9331bd1bbe19b27efb0078c55729be0f50e15a651cefc84 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f149e97e0257ddf4cb88b0ff81b393cae568134ecd35cf95a1d365908da1a6d3 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fe789bb083927eb2cd95f30b8abce1b86c1538566213800a3e88a140527af2c3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-41c2838b | sha256-0517edd3f4f042793b6d11ae89153c174f43a9245d41ef55f1549b402389e996 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-41c2838b | sha256-7427d9d87e4f1229f4f70abb3bdff8eaca681760cdc1d0ed96941fd2c5b9bd29 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-bfeaf8f0 | sha256-1f34c337682da6ab90e08fa8f8c578843af2f90d1313b30b2b6ffd49fc872e8d |
