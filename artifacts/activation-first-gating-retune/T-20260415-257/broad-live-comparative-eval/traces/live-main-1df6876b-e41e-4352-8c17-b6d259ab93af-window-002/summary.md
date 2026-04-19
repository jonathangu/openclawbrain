# Recorded Session Replay Proof Bundle

- trace id: `live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-77084407bbc4bca3a65c6d02bad38ef538707ae11378fb75424e9db3f47a8cc3`
- fixture hash: `sha256-1eb4b9074aa4e35ff1cdce5f3e7563b07cf55a4f769a5cba98dce236ab9065a7`
- score hash: `sha256-3c9323571781db3566b3efaf37edfaba986f6ca9d7b9edf8784a3bb6b4892452`
- bundle hash: `sha256-3d755350366e826f6c72e119c0a9ad08ddd3aefc9cca0634e9974b5c54954991`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0b5c306def10c6fde2517bdfa9ba5d43df83079e99c7ecff7a5190a715aeaea1 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-060ba4f1cce0c90db56c5c1600965e290fc80008798fdf8392eb9a70d361b52d |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-d0edb03f0f2e362fb01d70e197c53f19d09fd236db56be1e4bf98279dc7c27fb |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-0ef2c1c17055c05ca94b7eb798f4e420968284326a8cbbebb462673054e3e3b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-c97fdb5f | sha256-4b04be55bbf7800c29f72701b3c20b630cec33a3213df20b45fbd13373b5b112 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-c97fdb5f | sha256-23ef23d3d565fa3751c0e93186814b6196d7fb2de82f2c9592183dc7de071fde |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c97fdb5f | sha256-3e173fda5f41c177a2a79259d2a8fe949b4e52b3aa8becff03f5fc710cc4ce45 |
