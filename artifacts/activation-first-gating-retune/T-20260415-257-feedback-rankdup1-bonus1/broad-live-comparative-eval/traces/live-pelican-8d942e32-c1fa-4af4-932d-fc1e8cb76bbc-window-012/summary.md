# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-874a83098560adaa94c38c7c63cbf4c86efe4c86090d606bbfa34849e336a8c9`
- fixture hash: `sha256-b06776d862580d01d558132918aaffc22b9130c1387f99ca2438e1c6cbf7e22c`
- score hash: `sha256-3c26af721b3d9273dd69ab6703dc22ae7fcd8e4fa710a5e3abde095da336cd07`
- bundle hash: `sha256-a88cf793a30c4d7ea788c7481d29817f1655e47cd65c2911de70862c0dd43f89`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74c569833e5bf27fad2f2f842fa8eaa7d60bb320f690bc493bbf6c394f309f6d |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-44744d458c958898ed5abae1db3556aeebf0a01914d4061d7d5bb511db5ec0ae |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-c2677b58dffdb80400586d91190771c5ae288299aa880e3188aecbdef67f959a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8417e5e37c260c46853c2ca894370b649046bf5daba864b77f9d2e8a28317234 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3dfbd637 | sha256-4ec9be171d2fa451b552bf73ef4717479ff204e36a85c999a765530e6ee16d56 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3dfbd637 | sha256-e04f26ae5ef1e623622c275d9bf1cf19ccec9afc24051d0e8e55a8c151c8681b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3dfbd637 | sha256-4ec9be171d2fa451b552bf73ef4717479ff204e36a85c999a765530e6ee16d56 |
