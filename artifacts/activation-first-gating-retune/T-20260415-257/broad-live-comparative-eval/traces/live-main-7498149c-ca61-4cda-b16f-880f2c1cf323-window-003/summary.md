# Recorded Session Replay Proof Bundle

- trace id: `live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e725d27535b9af2607edfc55a55297b3d0ba2750ec3f143ed7ff11bf77f51432`
- fixture hash: `sha256-ad7016b7c1b3014d2314a0a7c149b979d45fc36075f6fdac2c19edc868ece1f9`
- score hash: `sha256-0647f8982657da9911f4e82a25ea8c378d8346905b0ac508d4cae2df51faffb0`
- bundle hash: `sha256-61345a1a3b9bff865de3f1e304fc791d39ad6501cf4acb3e3856a299970f17e5`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-23ccf4e617d8f12358b2ab10df7ef2356e76e977f4f8b850f824610003c0bc33 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5d301cf62cc1c04883180db2318cdfcb9cd690f2b7f9a8cbd04d3f86596ed5cc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-65310d2510031dfc47d7144b70a8470736edbc9371ad0de22199a014bae13c23 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4534941b | sha256-944fecc00f69d952a1e38cad33b73766e8f3cad4e04496a200c18e181e371290 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4534941b | sha256-b4606f94e5c5ff1e722557d8049639144d27ccc4e0e5b58482eb055561be2bc5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b0ce67f0 | sha256-3dff6b5414f00eeada17aa8de493073864a8baaa5cc52f3d2dd1f6f879f72e6f |
