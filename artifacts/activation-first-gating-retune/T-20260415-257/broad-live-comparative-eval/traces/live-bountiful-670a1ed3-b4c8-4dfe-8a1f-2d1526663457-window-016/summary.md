# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8d30d0b2ffefbdcd1e1a89d75d980761c51cd05c50f2c3cf1f693944186af036`
- fixture hash: `sha256-029c6b1d164f9bd1c4692f0184b6bb3b57e3ba2e59663e9c61a6962698d01e73`
- score hash: `sha256-bf221acb7c2c2c4587f29b20fbab38b5948fa191876cb4b68fed48c564772f00`
- bundle hash: `sha256-a66127ac32090b7558a9950ef271a8033b824d2b4ebe776d3ea21525cda8c86b`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6eb030a8259079868b419f4ae1a6c389dd22240eac5e867e187ea0fab1adf6c7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-281ddf36fe1c518bae673ecd425a181d98f9d00eed5e7bda8f3d3006454ed5d5 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-df6be93b28e9a5f976775873c373a5b3f6eff4bf174e17fb488904bc322fb13e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-c6c8ecc01f09e573210614338905656e6c90bdeb0166b48d9f2ea222d290726d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cfa2471e | sha256-ac00edbf7b50fcea8f648be6ac73e56eb1b3ad29f1c017235820885fac03b635 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cfa2471e | sha256-fa54e07a6a77cf40f5be4a7d72c8b3cd1e1ad9919362730b7e1a908a30986ff0 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-2289f25b | sha256-171b87010c2368757a457980cc8f9490469135b5f57f9c75fc9db148a5a95544 |
