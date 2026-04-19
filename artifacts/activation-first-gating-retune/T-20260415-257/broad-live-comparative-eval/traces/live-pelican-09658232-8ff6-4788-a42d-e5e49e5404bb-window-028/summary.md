# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b96b2a97bdbabf2a2491696460b8da77dc242a25c47533759e1ca69d544c781`
- fixture hash: `sha256-32449d86eb6b142eb11e1d76d43e4c37d62e87233bae5b870977e6a064fa97e1`
- score hash: `sha256-bdafbfb1f3df866275d6ed8c5ed7dc5083493c709e8afc735986d6a4859e4a14`
- bundle hash: `sha256-853ba8a8415a62200372eeb501e61419df95409c8586b946117a9dc9814c33de`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b14dd3575bcaf16e76897e36504d083be01ba320a2077714c9a7749ba84f112 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-619a1b97d5c282221931aaa4132a4cf1042179d9eff87268df6c69fedbc456ed |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-efc834349726f723a9bc1710ee4d739792f24a53655d2e97929544adce27d070 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-caa7028ad5cfe353cb59f2ca006c7645ddf87b7045ab995d7e4e6e5cec577454 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2d7cabcc | sha256-3e456e0391d8f1c6bf2689482506d6685e4316e66ba825b563be2236f3d59574 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2d7cabcc | sha256-0f97f15c88f1339976ce42bb83d88660f0b30b04bb691011542d78774021aa3e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2d7cabcc | sha256-3e456e0391d8f1c6bf2689482506d6685e4316e66ba825b563be2236f3d59574 |
