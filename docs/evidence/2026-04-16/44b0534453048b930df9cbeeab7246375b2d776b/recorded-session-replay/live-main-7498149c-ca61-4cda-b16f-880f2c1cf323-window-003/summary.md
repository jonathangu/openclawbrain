# Recorded Session Replay Proof Bundle

- trace id: `live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e725d27535b9af2607edfc55a55297b3d0ba2750ec3f143ed7ff11bf77f51432`
- fixture hash: `sha256-ad7016b7c1b3014d2314a0a7c149b979d45fc36075f6fdac2c19edc868ece1f9`
- score hash: `sha256-7453c0a5b363c7b179d1260777214e4d3df92050fe121a9ec120e086ab8b2717`
- bundle hash: `sha256-597de5598f63586106059029368124195c19144e9c657cb8b69164a7dfae36fd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-876d8e0d55a1dbf8e84f9354102b290654bd4fd9a97c3f26b37887de51274c7c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-97a9439b884aa274d1adf121c3dd2c34f6db740b06787bc130afe851973942be |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d55e4f4bd035684100f15c060e3ba73f2084cb9fa61d3895f3a47fc5509fada9 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8e52b0301147d18ee919a4af959ef19cf85052cf861878de2eecba1ddd3dd59d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1f931a72 | sha256-63eb029a0bac2669f9a1d5549a3455458dfc216d8cba8c97df294853b682ff61 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1f931a72 | sha256-0e9cbfaaf77abe7bfac842c3242f086d954d9d697cb08579d87a5eae13d18bbb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8b2cee47 | sha256-45189611823234b14a8dd63b81cf71fbcec49773c25a3d6c1a25133d66d0bff5 |
