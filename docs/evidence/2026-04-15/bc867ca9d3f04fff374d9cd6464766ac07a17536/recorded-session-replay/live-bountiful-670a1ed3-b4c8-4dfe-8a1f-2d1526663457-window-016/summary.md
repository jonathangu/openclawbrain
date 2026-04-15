# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8d30d0b2ffefbdcd1e1a89d75d980761c51cd05c50f2c3cf1f693944186af036`
- fixture hash: `sha256-029c6b1d164f9bd1c4692f0184b6bb3b57e3ba2e59663e9c61a6962698d01e73`
- score hash: `sha256-a8074f31d68b1b7e2cc99d63bdfeb3b0d955449fc044d1f2b795b54534ee6a24`
- bundle hash: `sha256-e35b7a89d4a69352d354f7a5eac8186aed82c48adf3843386972ad7ac1461a5b`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6eb030a8259079868b419f4ae1a6c389dd22240eac5e867e187ea0fab1adf6c7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-19b8e0bf0ee55482b03266db35cfb633b2262e9519154fc44be73cf51ccb34a4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a5f980bce0acb25ace17d1621af571108ec66a31d3e5685ec829a18593301d54 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c624a088d02e4e77f96a894d22eb578acc4b8e5e9c56067e8a9622f94b2cf50f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-451e4487 | sha256-e2a2f5bfa8a2492238b3816a8c199bac753f24131931c3b0bc186ae1c961b197 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-451e4487 | sha256-eabd5b57a01224b1f16703a8e40948a966399147bd2dbb601082417a8b3a0bc1 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-451e4487 | sha256-e2a2f5bfa8a2492238b3816a8c199bac753f24131931c3b0bc186ae1c961b197 |
