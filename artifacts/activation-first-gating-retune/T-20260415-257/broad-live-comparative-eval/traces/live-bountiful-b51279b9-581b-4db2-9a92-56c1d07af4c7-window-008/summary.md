# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e05f1753c6926f7421800b8c31b26225feb7d252c56da262406bfbe6a5f19442`
- fixture hash: `sha256-dfb3a653195516c3dcfd429e73c49db1db57d8e9ca226f19c9bbf361b6ec9f1e`
- score hash: `sha256-10856e4224fd8ba7ce1e6306703d3debfc1619e52cd83c8e8883453575bef20c`
- bundle hash: `sha256-1505427d0c55d70c6a08d7d6b786d4d0644a18acefe600b1147ff9516b2f00b8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-560b7797ded1325b1e9e670019b80d08d942ad37f4650ec817919f6cee20dcc0 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-30da6216df11505f2144c2b73db7124afc5d9227b48892b8623293e149fa9432 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-29460711443de0972220dfaa9f98aa7f2394dba85c5c56f064be4416210ddd1c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-84d71a1846cd8b49b13b21ef737464bb2ddc430ef9135f0bd65a0451885f5dcf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-e4188736 | sha256-2e6100da56880973d7ad1e7db94d06bdcbd07841a07a95084262a2f8cda22e16 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-e4188736 | sha256-423ae2375e28565823be65cf23215f27b9af59f356fd7acdd1fcca30a5be9a16 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e4188736 | sha256-2e6100da56880973d7ad1e7db94d06bdcbd07841a07a95084262a2f8cda22e16 |
