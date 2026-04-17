# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-62465478f20fbe3d79873886b302e2840857c387394ab96d28f59053370cabaa`
- bundle hash: `sha256-ccc1f606445c03708828f07e8b98b727c0ce64ad205d038540b68bf7683a9246`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-391666dd2ba1ef232d5fe1d6e55fbd23e14e469dbd3d9f9daccb34ff97397363 |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-20fe5ecba20e73b326852c517fdf5a9a54af031b7da5b70bd2b09aec6659ed8d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0a5148deef1af755339c0dbaf1710748cb0a930760fbe0c2a80d252a7614c61d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-c64e7743 | sha256-a4e02fcca53d640edf8e9d4ae4811e4323fa66b858e3f27efb637283ba7e0415 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-c64e7743 | sha256-8fe7ae8ff5824b5233e5c05990f5a30dbc195844b3f9616559a09fe082f36c3c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0609934e | sha256-95e2935b6b15fb105cfa1a4ec79b30af0d628b9103f81b7936fc21e63a3148d1 |
