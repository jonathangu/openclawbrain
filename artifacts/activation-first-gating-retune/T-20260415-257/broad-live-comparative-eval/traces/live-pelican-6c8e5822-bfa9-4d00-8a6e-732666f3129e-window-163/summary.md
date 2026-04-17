# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37b9967646ced8e1a7e53e66d95e96c0d5cf9872e9f6cf5f223ff75c45212fe4`
- fixture hash: `sha256-e5562aca0bd9165edb9d4f0591f9dae6981c5299e9b8cff4453286d3a3e6c950`
- score hash: `sha256-5b4f2b202797756bd81686d27686b0e7d25727d00953748980a22ddf3935f36d`
- bundle hash: `sha256-1fd6ef968d55b032d2de5f3c72e81e9e688eb2418be6a9c960300328c0a0f353`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e59e0368806e0012160cf4b2dfced7c5e08071a2c01bb62268694e031a82feac |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-082259a87c6191207148ea0073da82013d6cd1c88356394a1e21d3fee6af9184 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23d29af60d0ec39e1e3afb39a18ecda31c02f76d4e65e952aa56ef96be02c56e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d8bc12aad1d94180b155e42f4d2ff78b96b36b9f82e1c226da7ab7509a488fca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-717f09c1 | sha256-fc0bd9c408722c525a55aa09d4e1d00d19ff9ca6f61d6e0bca6844290368f234 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-717f09c1 | sha256-2003bd1768d3c110a6879a4ca1479e02302c75d2512cf461d1b172fd7ffee935 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a25c5d76 | sha256-120782c5a5264c37a2b8d410788ddab11c3670740efa9a774b9b732dc2c55c3a |
