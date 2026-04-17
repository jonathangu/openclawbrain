# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1ec62e8076ee4d1e710644be210d5ded13133f83ba7cc0a283a8ff2ec6e4b13a`
- fixture hash: `sha256-208011a3d49bd10b0f228ef3f15f5d25a591b8469fe6d29ce8deec0246fbbb48`
- score hash: `sha256-a34888210d50f9ff47d36b9655d8d8c091adff1d9fde246968fb5483da237ab3`
- bundle hash: `sha256-1580a3bf3684183e8d6d58f1eaacf321508891d0a2ec44f9152380d40150eb57`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30241ba1cbd874d0509ab1e29b9c021ef1eb69d9f017747456f3594de63d356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0f4d50d9c2f0a3d79f66ad15505c1ed9f7fa7a48bd0d5bde611d19f4116c1db0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-51835c72c7ddb1572fccebf03b83f5e9be571100d9430f983e084d0bb3b11217 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-664a7460d345b6f47f343b793068e19a96d4c509ffb2a22fdf572c325dda3236 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dc3bd384 | sha256-6795e17c3ec6249f05985add7303d5ae2a384bc765363fd9eda2ba493ac962f2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dc3bd384 | sha256-86f31d2ba4b4cc671e09b696a5d497fc459b362c25c81419ab3017b32c032b58 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9e5d2297 | sha256-900e6faa7e2357ef3e226959169f51fed69d57252d907a818489401666218425 |
