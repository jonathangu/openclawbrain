# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8b83288abc1a5c66a218574e9a089abcfea75ee1de4f5813fd07c339a4e34fa2`
- fixture hash: `sha256-d84bdb541f6a2d5c8236abca3a843aa21a0e1c20f003d0fc5eb1d79b307b698e`
- score hash: `sha256-38781006fff16ab6ccb41dfd7c3d0730aef475766dd210c3e8b8c2cad3bfa3be`
- bundle hash: `sha256-e2f997e652ea238e55618deb65820c8835439e346a5cebb0f54a073f94425074`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ad0dcf523c4d76bf7e5aa9a9c949e660e04aa89d0cc57603f9d8d3b2165caa4 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb3c75e7e9ed6676f5f5e8fe6b9949684e79a6054a5433fdebdd48c315fe6a1e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c6746f951d82a96a26d4a5813ddf2125c5848c055b3a9f0981050125d12f785 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7fa6858613424fbff853f5826969fc35e112be6c2dc4d73d60896fb8550ec502 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-80bc0de4 | sha256-9c9125d1c540fdb30bacd96bdd80188177bbce3053332017b99f1227514e2a04 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-80bc0de4 | sha256-ec2f05b8a343b0b282249d6922b459ade4217c8322b6400db571dc2c9afd59e4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-26143d53 | sha256-b967b2538b02a289fff9900cc46c62f81d6969944227a67e1249558cb827f120 |
