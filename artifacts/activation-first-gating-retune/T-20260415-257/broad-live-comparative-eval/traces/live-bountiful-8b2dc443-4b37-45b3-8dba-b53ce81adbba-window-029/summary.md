# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-766d9b6ce430d9d07fe2ff3297e9849fe05332d7539d3d62db1cee2a9f89081d`
- fixture hash: `sha256-21e8a90c2dad8ab78ca636bf0f382e5b550e2af76a7681917f1773769c731648`
- score hash: `sha256-7d194e42973966896568423401cf1df873f6df72b9bed54c47aa62b873cef76e`
- bundle hash: `sha256-b6fa430e5032c54392a373faf2a9ad61c4bbee3ffaf0b4801e0372fc1498312d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8021a8424a98c9c0ae913d23bd911fe66b4179fa226e5ae4873cee34e53cd89 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9fa40a56e71edeeda25333389f538e816c81234e206a09d39287c1f186b104e2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5a32d313b06ca567b37432c1be498bf32cab2b853821105739c77db36f88767c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-20f596e43cf96670abcb850f7523ca31c871138d5d3906c028a97bdb54a4d133 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-82c50afd | sha256-1c1795ddce7cc13240a6639577ad45a34c9de3c2824dc7664775bcc098f586ad |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-82c50afd | sha256-90d54dc2f6a751be0d4bb47d287b45e3e92bc56821ecdbd71af9b41ac5784092 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d8fd5f70 | sha256-85ee89f79cf26612fa4f66afffa23d1aeb58cbcb0a080e650baeabddd1fd5ae2 |
