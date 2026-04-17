# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1cc116d5a5a3e4268eee5081d6d597a83a2afaebb6c2529b01952ad2f45437c1`
- fixture hash: `sha256-ccd8a0f1240cc7f92941ab2c1ede0327e4ed0a420f6a51ec4c81e0437c7d59e2`
- score hash: `sha256-944165d127754e4c1b5e285c0f012793a7cdacd68a806900564a63507cfbb25e`
- bundle hash: `sha256-13ca61137b3ec6a460e1cddc596d511dac3968862f611193383da988894ff906`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a97ee439ef356b4483f5735f34054ec24021480ea2dadec6ac22262eafbebd17 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9bd3cad9bb8174f6d1ed650a86c8a98dc2521f5f8f1b4b250a96b5f81c074384 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c6652647e4393b03087e7c8e10ce4a63daa5f659ea8d051fe876f3b476377ba |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1b91abb55f39ce2e952b2aa795e4b548c776c5740e441f6baa2fc47fb860e035 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f94000d | sha256-c7f384400d4d5d868d4534939dac5a268f269fa099ecb4f7457ef12ffc1d6646 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f94000d | sha256-77f1509be8bc47f1ccf336a493f8690fda51180197a27c51577be4db1d495b2d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-17fbfa94 | sha256-6e5c096c3543715101550b704611fb405e6612fed1170355763f5b9703e5733a |
