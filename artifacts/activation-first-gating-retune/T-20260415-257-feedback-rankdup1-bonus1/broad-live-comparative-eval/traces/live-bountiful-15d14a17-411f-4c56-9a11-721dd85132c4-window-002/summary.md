# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d48635c1ffd88f3a117b615a76004d5e367f3ddb12c33db64c7bc064203d9b95`
- fixture hash: `sha256-8c0e50ffbe18960ebf818512a9b376865f8811b20166fbd695c968bf02943a6f`
- score hash: `sha256-3471adffaf4cffb0ef6ef34dcf6010139001ac80ef3489ff4cfacfad1e9985d8`
- bundle hash: `sha256-8e57ca4274cc26718da02233279580b9c92a055bd1155fb65f66e866a937e6dc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6f313e5fabd4dcdbe02c394910f74b656656e1556ed3e55025a3581d3065dfeb |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-50deb3852eb4a88a8814ba59080f08febc0bccb23ad0e695a8b69d250affe3af |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-044fc75ef8ee1dcaf07c29d9466fe0b256f764ade0faa897653edec801a309b6 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-3c1050ca53f509b4c8707cfdc08291932601b0aacb2db12b498937d2c4952923 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-30b87ebe | sha256-3c11311dabf383e50469efad625275b3be94255fe8d01401370483633907708e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-30b87ebe | sha256-3c11311dabf383e50469efad625275b3be94255fe8d01401370483633907708e |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-30b87ebe | sha256-3c11311dabf383e50469efad625275b3be94255fe8d01401370483633907708e |
