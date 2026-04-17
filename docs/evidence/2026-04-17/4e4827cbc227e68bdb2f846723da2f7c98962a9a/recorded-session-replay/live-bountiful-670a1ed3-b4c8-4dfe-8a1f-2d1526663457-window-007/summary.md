# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-829093e68c8b369222680a9ce88928380b9f7729d760e59ece8cf8d1e776b82a`
- fixture hash: `sha256-31ba926396eebecc30aa75781e7d614cd75f3d45744f5fc68d2426d0829db138`
- score hash: `sha256-9b702b0880cebe66e6e2d08eeef6031c2ec2b66593117ba1d64b2dcf8a53c63c`
- bundle hash: `sha256-ba2eff71d2405d77839886949a0320fd899487c551f36a3c99288d96422833de`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8d38776b0b580dc292e4970ed98776136ff0d2acc01ecbb7a8d527a0c51a84c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8c815882da94964cfc6abfc97b6afc1f0f622fdc8f9354c6f77122c272e64a8c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a7153f9818489df5cc82c40089da4a31074f9286c6f233a00eda86e1f64482b1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-75bf6024a2950d24d151ac269effe09758a165aafb5693e077bf544d69cce8df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b4da678 | sha256-3b2f83a8c33f76faef398fd14e8920b66875fb65f69d0c0836f6d5abb5008753 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b4da678 | sha256-182210dc550bbb0c87eb127dea87d1a7856947c8213f1d342ba528d37c17eb4d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3d7cc26b | sha256-dd1f009607583547f88b0ba29508178c51e9d543ef6e96f62ebdbdf8ab68a5ec |
