# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b29852ddcae763768818b925bfa513423dbca5c8ad934450c78f0838b90cfab`
- fixture hash: `sha256-f828a5ef63881667b78ea5f5530e5417bb5590176f57bdcf8c4590150136788a`
- score hash: `sha256-1d21eceb54ac9dc3787fbba5e25a9465d2e99571ce1ce1a429946dd9967ec818`
- bundle hash: `sha256-1504c950def89a30aaa1a7280c519d2fe548780fb5eff06aedc947e12d4f028a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d23197d4b519cff22649347398dfab9ce049fcf294afab672a8b41fd8ebcbbad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5af04eef1c3f6707e495e89c40278f8b7b46ce9b5c255b18ab427ca09c1a9843 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-67bfd16852b13eff77cd7c4216d4f2743bacda0c9c7e39f4844081bc9038727e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-38514bfa317f0d1bdbfdcedae6c7ec0364df5ed149d68e9c62a11412491433e5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d28df021 | sha256-9712c9ef1acab353738b911339559988b2cb461820f28065e4b3c4d8ca93ccb0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d28df021 | sha256-0c901c48d1b6779fe15b68dbcf4f4bd2c7509a51e8c43e85ec7f1a0516ca3d0b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-92915262 | sha256-d96c22028da1c551908f4d0023279581d23d78c259a7c1cb54eca209d0b97267 |
