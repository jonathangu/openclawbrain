# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f80bd1bbdcfa166ddf0a470afc83601b492c4563388d58e45f1a2c05fc2e95de`
- fixture hash: `sha256-6351998b89e93fb758f480838cac801b256b59b498a256a0fe6d16fd14c2a7f1`
- score hash: `sha256-ea70a2eb7daf31b8268ab77563b5997f6f6a51aa6d4d6295d91d14c2af4245ba`
- bundle hash: `sha256-31dd6ab85118c7bf06001119208c4d667b55c9ff8b861608424acbfb45f365d5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc854933a918fb32be2808e49cd62ec70012cfa090e09597f270649ed2f5446a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-89b6840cbeb574387403442a5670d29783e850b18c7ba12df33e9e6afe46d87a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87d53e47218ef44785fda723acf36bb105a582a7a7ea742f1e7094fd9f51bc46 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-565b4286a2a991160c1731dc9c468c26e110f92d556c3d14cd53120ee964ff14 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4211079 | sha256-033413ce7d3d2ba0ccfb16141efefbf9a0609ab4772c585e3422c650dd29fe79 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4211079 | sha256-a540588a8e82b6f866cc245677093d87fb075245ee0ec243ab1bd615bf50a7ef |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-60e9ca86 | sha256-290b113518659b113cd0e758e14800085418dc6cfefa626d2d309f7e622ee330 |
