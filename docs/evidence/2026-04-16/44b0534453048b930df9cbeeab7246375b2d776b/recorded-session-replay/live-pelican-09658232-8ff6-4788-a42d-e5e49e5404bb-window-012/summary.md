# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50bf6e94091f3556fa81577b5a708e4425c4e417b6705bd10df603b7966e593f`
- fixture hash: `sha256-18f79e8bc777ea9555de29a01a8501d21c8ddf1c9ea32bbf589d49b4f4a3aaeb`
- score hash: `sha256-ff59854a3650f1213a8994fd88f92ef064e446965de9a2a9e50d3396af566f94`
- bundle hash: `sha256-90290fc119d5b8c7225ba159bc6373d487e04ebdaff41b820bf8fafd40293e76`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12a31c86174cd8cb94081745ed9b01e9e8efd75760d60dfa60b0b81778821ef5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a06202aa795807893b3bce462e9f273d3d0cfe1490a176d37c54e7ed5a9f7ca0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e4bbf8cb92563e1e8b62a2066ae3b02f7a5241f48633d0415d275817f2093fb |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-4f919d1f1325df19d9bc1f4e204a45c018c6552673a1b77d031cfc95f03e7b07 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5aac4777 | sha256-5140fb24d7b005c98b4c0b889a356ec4d9ca720a454ee256cf37d0e2476999f5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5aac4777 | sha256-c6fa73b31acf95667bcfdef47bc09ae3a8c2183981d7cf10e529e78a3d696f96 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-294734bc | sha256-1a5359beaf928f5c6515c3e6da9fd46fccab6c0f4a3d011dbd7cf70a7fc8007e |
