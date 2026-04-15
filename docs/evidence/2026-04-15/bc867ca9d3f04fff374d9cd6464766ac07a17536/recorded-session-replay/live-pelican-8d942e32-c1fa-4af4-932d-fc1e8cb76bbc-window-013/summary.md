# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e8abe8bd791e7d6cf823eab880acb642edafbee61d1547309c32e0509f5a12fd`
- fixture hash: `sha256-55ffe1baff231052090ba7af248a8c8c581b0ed9688d4757d7043a08a2fcb4de`
- score hash: `sha256-294b683d8cd50741caa84d2652907077296e14b1e13e02b1963bc2644e0f7fa1`
- bundle hash: `sha256-025c12bfcf1b85b675823ee258815d3781188948b63b62ad0207ce6777b92499`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1da03994a3e5454931ba1a5c62fc1691a06d32d29326ec5baedfa4f4b490d130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b9e7424dc1e9234f84466d1efde76ee0a96cb7ac97433ce71a78176bf3035c71 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c8971cd7dc9197747ccfb7b1c7027875604b18bac56ef0e43b9e6cd63ecbc3c2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a959e87464aee6a51b782e14d9ea2817ab80d86c026954b41b835d6599ee7f6b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a915cde9 | sha256-0ca97d2eb769a317e5485a6da1b6a0167392eacdc4501d97121a8df3bb935856 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a915cde9 | sha256-85c9c3e9cb116b64117b25030c90edb59a460de715cac7da97c34214347a0263 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a915cde9 | sha256-0ca97d2eb769a317e5485a6da1b6a0167392eacdc4501d97121a8df3bb935856 |
