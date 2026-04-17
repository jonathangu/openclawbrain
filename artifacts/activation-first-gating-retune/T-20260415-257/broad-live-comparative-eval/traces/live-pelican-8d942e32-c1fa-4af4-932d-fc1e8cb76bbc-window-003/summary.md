# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9bc9ae6eca1afa6b83b6e75670188dd332e3c4f110971932dc47f6d6a315c1b4`
- fixture hash: `sha256-802793623b4c122b10687187b8ed29a08f9e42bf4ed06ad0911f576e3bb3e669`
- score hash: `sha256-e43b85c42cafe6af51f3822789b329de4ae046142c8ead68c29aa449877e9dd0`
- bundle hash: `sha256-81ed7f691ebe60668e3dc81ff55ee437f35eea54d4d914fa4f0abf632912a9f1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c6cbf75055b2f2066145c893826e751aed7af61508ee55203c7ec8985a9cd38 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-72530f66c52fcf938f485dbd9ec345ae2e3b4bacdf9eab95169fe7dd3825cc2e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e5f63a30922c03cec6e6b13d189a9fa343a97511edd486b18ea940bdde719000 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e8905f1adecaaf961a3300cfa7c6da6d892ef667849b466852a72270a57f0314 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-06ed1fed | sha256-bd5bea7fa558ccfffd3f5cd41221cab8458609e5c98afb247e1513b0097df0f1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-06ed1fed | sha256-e07aff270218e463dcd21dda22a9118cc21e1c0b01511700c793c4aac3df2dd2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-35d01336 | sha256-c6d945fc9b256fda24e94b32f5177d5009ba641d963b0a2a46fe74bb0cd7b344 |
