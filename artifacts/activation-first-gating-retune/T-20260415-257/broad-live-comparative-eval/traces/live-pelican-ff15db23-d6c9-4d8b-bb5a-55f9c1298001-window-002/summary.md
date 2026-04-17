# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9abdc2f8435606514daaaad4927f60e901a9d2b092eb5d39df77887ebe5a304`
- fixture hash: `sha256-6e857ab9cb3ba1ec3e0f72cceabb24485f23daf6db41d61af726b2888aeb0f66`
- score hash: `sha256-03fef889ac87e6d76c29c30473fef04cdda6e8ec107fff8d7e0df140abc216ca`
- bundle hash: `sha256-e66acd7796207f2c12f2454382cca1f230449f4b3f92f44dda57846cf4f055a5`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-865378fe979515e6fb05b86bb93e571f4e3d4c4ed17ab843485b9830a42b2636 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e3b58d9cdbdc8c6678c9fa61baabd54d0570ec242666d2968400161dfacfc9c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-880be0734ff79227640583ddefa4f52a9e67406949911aded6a747041ff11cd0 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-34adc1efd9ac99caf0d9e5bfb6c3ff7e104ba3dc5543833951113e11de839395 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e86d9707 | sha256-39a3ebd4dd226bb35ad3f8d05e4411441c9ed28ec308a9b0df22e4f8d34c6bc3 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e86d9707 | sha256-cf076f258a20fecc1d703a1ba7ab28884515344f4d5ad1b32ef8a1aead902d7b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-78a6ac76 | sha256-b2b3b3c9922cacb62fd799e8d4f8394c38a1375c1f941cd33d9e3fe153146b26 |
