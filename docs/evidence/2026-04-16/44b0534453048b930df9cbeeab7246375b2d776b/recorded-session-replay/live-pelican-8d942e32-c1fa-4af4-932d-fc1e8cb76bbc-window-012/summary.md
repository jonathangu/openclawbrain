# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-874a83098560adaa94c38c7c63cbf4c86efe4c86090d606bbfa34849e336a8c9`
- fixture hash: `sha256-b06776d862580d01d558132918aaffc22b9130c1387f99ca2438e1c6cbf7e22c`
- score hash: `sha256-621e5837b1c7819ac9d66d46800b87005f8e9dc2759a845650c08fde383984b5`
- bundle hash: `sha256-7fa8ba1f7a7f798d2df79eb395f204a284afead24f318433e3007958d606d0a8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74c569833e5bf27fad2f2f842fa8eaa7d60bb320f690bc493bbf6c394f309f6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6c44fa24116a4bc9e45e46529bc616799d25c49267826abdac8e28450e08a090 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-49028b1ba67e688ce0fa5e8b6cf22c28eb938b88794e13a628db68b7d94c5ca1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-4975f2a2b84ec06356fecbccf708d6a9da5cb6da7f1711be5ecf11d0059f77ab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e574e018 | sha256-e5e7ec115e4b3c91bfff5418ca9129de654c8dd1666ee0920f6f68a51f7ac8c2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e574e018 | sha256-1bbe17849297260272cedd145130e24b8dbeb13bd79e93fdae7c6813729719dc |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3e265019 | sha256-f26c51a69d55bc5ffe666826bbc2ba78f7724b8945080ef7f7df24586ff97331 |
