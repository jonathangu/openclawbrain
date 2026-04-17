# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e0cde3a25a3b093a3328111ec29f970fc82068378dbcdb19446f77be2e4c1e`
- fixture hash: `sha256-a657cfcc3c13a64972df27ba9b34b582252db2226ee691420ef45e3b6a2bad38`
- score hash: `sha256-9e437fd3d68db85b71e00c29acd3e22154b42374a0f6a7b0bc71c34b577fe23c`
- bundle hash: `sha256-943bc6202922d6745dbb95777ab5a47d78151e1f4269da118fe1b9c8bd43a1ba`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0b3855bdbfdfdd31f2cf7aedce5b8a8e42a2e757ba398c67ba975aa86dd21ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fb5acd254f5d643f621c6adad6876687e48907873d8be82df4ebd3d130b92a43 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-628d14f8d9541dd6b8b6eb21e70ef2c87e859ba8c8d47442684e285e04ce9404 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-eef1cec9a5ef1c72bccbcaaa5a7624f0cd20c577ad319d05117d5e4b139719e7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5ce03fea | sha256-4528aed1e36dd7df6bf20f7a6f67e81fd65426ab927de2721ac2170d87204598 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5ce03fea | sha256-33a2b497768f8c3f84c26ac81c5a3ea7f206b60a8757b43fa953f37bccbe5ffb |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d9d13d4d | sha256-4d655d88f1130c2858061766a3ff1d87b79bfb3b83dcda0055d18b9db945c22c |
