# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f80bd1bbdcfa166ddf0a470afc83601b492c4563388d58e45f1a2c05fc2e95de`
- fixture hash: `sha256-6351998b89e93fb758f480838cac801b256b59b498a256a0fe6d16fd14c2a7f1`
- score hash: `sha256-bbe24d75b86ed19349b9da87fd4acdea0be88e94bfadd95d994791bef6fe8e8e`
- bundle hash: `sha256-08544ba5b13cce8be46c0d09435bdd714ac021ed5fa10622322131ea13ee6f11`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-84224a8217028a2b33f1f6ee1079f5ff22542f1f7579761360d879876a461cde |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-059fbb41dc739a85f1307c36a89dc7c0f081c99d5b8f05026ef523bb8b6a1a62 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6a144629c7cee300d0e03371ea23567c39798adeaa6a8562fe1428f1da732388 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-28d06190 | sha256-b4ab9ee0aaac51c88d52d70cd10e734616c6b552e7441f86c18e698d06a0c2d5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-28d06190 | sha256-59439fc5933f02dfa773ce3eac3d1633c782df21fa269d96d4514e7e3cf584c3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b5991b9d | sha256-f85495991b3ad6da6eac1fd3e10858e846563fd0640b35dd41469d00a1a52a19 |
