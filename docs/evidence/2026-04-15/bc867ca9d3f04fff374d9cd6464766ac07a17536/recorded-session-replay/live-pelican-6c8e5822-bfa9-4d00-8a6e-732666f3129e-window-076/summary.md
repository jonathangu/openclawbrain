# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f8a152004f1762e8eed1ecc9dffc7029171fd911a8be7a9ecd27602349fd8ea`
- fixture hash: `sha256-fef2416147c50461e059554c89ffc13514d9838c25717ac0bb496af10eda074b`
- score hash: `sha256-18e330e61dedd56ca2b6d31890c633cdf33f304341596cf2709d05868c205ff0`
- bundle hash: `sha256-4d375631a16bca882e99ad584ac834bdbd7690b4d2f4a2eb36a005a9898649d5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-510726e0c2b1103191bca21eb76122edbdd44953bed132c7c6febf953ec52703 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d55e3de0812e514a923a5a749359bcc5b56e720f809f622be2a1df7eaee4c415 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1eba4de55ef597aa079e88d41e91489bd6068018cf063e11dbd53f89b29bcd68 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e237e3ebfedcbf6d9323cfd7754dc1766009de30fb1b41df3f3ab898daa8d4f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-01a96714 | sha256-ba9d1112ff8cfd183fea4d47e2fbd76b02efd3d5f8b4ed053f96a0699b517dd4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-01a96714 | sha256-ba9d1112ff8cfd183fea4d47e2fbd76b02efd3d5f8b4ed053f96a0699b517dd4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-01a96714 | sha256-ba9d1112ff8cfd183fea4d47e2fbd76b02efd3d5f8b4ed053f96a0699b517dd4 |
