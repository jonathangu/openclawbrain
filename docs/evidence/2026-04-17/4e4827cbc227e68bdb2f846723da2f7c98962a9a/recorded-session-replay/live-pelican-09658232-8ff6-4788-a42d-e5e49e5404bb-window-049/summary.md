# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049`
- winner mode: `graph_prior_only`
- trace hash: `sha256-10f30fda1583220ffcb0e13cb73de4976d5f3f5f0f058e8e816ab9eaaeb4bc0c`
- fixture hash: `sha256-81aecda5857d0ab09faf0a56bf49fbe289e64582b0578df3f1535d5bf05ea11e`
- score hash: `sha256-b15f702ed078f3797cb14b871e126982855ac75bc0eba7ffaa8cd1cac078fb10`
- bundle hash: `sha256-23c489116c03224c56b90bdaf9890f133401154d1f8dd658e1d6ecf45fcf5220`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd49736decd703dddc6036cdf0bf744059f6270cb8728fb209a65a281dd21058 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea9af2265ff094eb04a2e7ac1608e080957b6d7eb3f0373c2e358c03b9e3a805 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb7de39614831c5a8f3153656520c1ae4a067b5def299b229aea9f2abcbda99e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d27d1ad8cf10fa074de5baaec30b736a57e0e09a647ebea72156f56575e7cdeb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-38c17ea9 | sha256-74d9ef84fdcdbede5fdf6b40bb1ff31fb3a823adea91f23c0ccaa18a94bcadcc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-38c17ea9 | sha256-399bda20bdfbd684f29631cc1454fb680066fc5be8408609efd8b802dc754452 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-96eb5324 | sha256-6a307201e8dd41211f97d80291ac4c685fcfe52e52d02c286658097373849a18 |
