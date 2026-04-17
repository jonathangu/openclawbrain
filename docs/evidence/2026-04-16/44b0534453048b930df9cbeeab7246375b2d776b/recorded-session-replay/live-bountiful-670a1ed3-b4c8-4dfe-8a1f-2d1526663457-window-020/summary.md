# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b202c1c438845d3c1c73ddb7c1ff7926a10fda7c3a64127ae541d469c9475d5`
- fixture hash: `sha256-b48968b0fefff768efffea4ced309b4343ca39a6dbbeda150f150e0d012ef675`
- score hash: `sha256-ef420ae79a0aff925b41ebd6cc27741ce9282cf9dfc34c0edd15340c8b34654e`
- bundle hash: `sha256-915db855f9ad0e57def7412cae2f21ea38c6e5d6e3400e891ee67961c08bddd0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-07841a59820286934b7db3a291f9a2a056f9291d9bd4bd106e744c3a6ac3c6f8 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68a91b1e9dee31d2a64410ec16fb8dcf6d0b1f78a1ed2ce733265ad50fc57035 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9ac418532924a5337bae200f268ff113cb5a797bc8217389f252e26074e0deb |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9db22f266ac4ae945f31f572d688e5bd93d67b9ed6d86b60978b3f826119d1a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dab468ef | sha256-5fffe7cc75083425dfaa25ef219f68805ff2003e1319955ceb5018a478b80227 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dab468ef | sha256-30a01d9cfdaf94530e40b7a8f823eda393a3e297a7686d2e3cc2ed3a8a3c0cde |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d9e4c4e2 | sha256-f35e2c0973e910eaefbb61c43a743e33a7a11d189a7ad23aecf757015ad1d722 |
