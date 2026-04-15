# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8470650c25739a12e09f620484c19ed76e535bf12ac60fa5bc19c4d9e71da263`
- fixture hash: `sha256-1a290f6c39ed84b2ca073e21a57823e82667ab7d1408676870645010e286d76d`
- score hash: `sha256-7d8bd12a944320d32e5eaa2a7cedec9a5d3de80fe3fff6ed682a64a98e20ea00`
- bundle hash: `sha256-7d7889b96a7cf6efb304ccb7eea00420c711eb0434a6c55635080bd54b38a566`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-575c5da35d27014ccbfd8fb043d25f84e1287b134d1db92502cb2f005c370afe |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-19ef0248abacce647c01024dce45b22da1f72c40cea9ae7d486c8b419a563393 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b446e8b64ca3d01a93672f7a3b780a04bfeae6851e7b8a16a31143c0740e8548 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-73c13d8e5e42a3d9cc75efd1a002c6dd77db870c02225c4b9b2609d5e68997df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-832b58d9 | sha256-c114ada2c05547dbbc7e8d54ea9ab6511d41ecc6871902e448fa7285601ffbfd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-832b58d9 | sha256-670c920b26d63fd2a013d465b22544cc77db0e663e26dc03e9d82839deb0a64b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-832b58d9 | sha256-c114ada2c05547dbbc7e8d54ea9ab6511d41ecc6871902e448fa7285601ffbfd |
