# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-570958465e5589f279bb28e850af48f0de0e358122b512d402db3214c7541c3c`
- fixture hash: `sha256-06808b26154de9486de3e390d83e02d5c54e1e0ca160f5f4c88501af04825dc3`
- score hash: `sha256-8257b3ba2412fae4b4521e1303050846b6ffc571db07f10cf613596eb7b9db70`
- bundle hash: `sha256-b3dd82c46efa56f59b01e7a170e95c5373f22d1cbba3dbb0e41fc16370ec1ddd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd27477bdd5733b8ef83edfac9b06aafa0bfaf3753550669b2a8358e4c2d729f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-81635e818cde1c2b6cd60d9690a94ed5f3fed9ed567ff3e60113911947760a75 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-11cab471a17f47d1575fed752e5e29555f5478759c04939002edaf0605cea0b0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-36cb91c174e4bdb6650e0454c9116cf9aa19a634e7e2918f4f36a9dc072e50b7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-53de8651 | sha256-bc5b867df75e4bf808072cd42727f582aa60a5d5723258fbc7289edf845757c9 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-53de8651 | sha256-ac29bd08b2331f7bb3b1be5141faff7cfbc02be56189f615db8cbadc5b1da15f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-53de8651 | sha256-bc5b867df75e4bf808072cd42727f582aa60a5d5723258fbc7289edf845757c9 |
