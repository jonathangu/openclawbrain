# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-07e10f6820bf810e1999011ea58316d9d53ec99aa0ac7473d30a2c9a79d153ae`
- fixture hash: `sha256-54c4d68b5e528e2dc7ad50c599fd75e1b659a972d8d4c97376e292a3ef62dcc8`
- score hash: `sha256-37d6f9d71ed73a19d973ac82ea8da89fdb04d52e3080b66bfd69be3ce9b5bfa0`
- bundle hash: `sha256-abc337f46991c499e5c820cf90da4388d4035bfada55d7ab7191d298a88ba062`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-430dcaca40205cc8d42bfba95521d8acee2a6e6c074b542cd0b9a2d9f1547939 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-d5003f508770b5f16ad4b86e1422599cccd22518f45a6140ad3fe1507915a58d |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f7ed869017a1a2e4ddc994415ee129e1423c911088450cb2a95373ac1a80a609 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-17d6b8c6f5bd328fff5e4570aa29e6ba11f057195621ca0177dba6f1f55e2a2a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-855640cc | sha256-8ff6ae69a67083a3cc803cd1d87525a1b25d22bc0ee3034c70f884ad66b705cc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-855640cc | sha256-cbe81970f89bbcbbe01f618f3c2a638c71aabac7f40924bf801bd971f02c861c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-855640cc | sha256-8ff6ae69a67083a3cc803cd1d87525a1b25d22bc0ee3034c70f884ad66b705cc |
