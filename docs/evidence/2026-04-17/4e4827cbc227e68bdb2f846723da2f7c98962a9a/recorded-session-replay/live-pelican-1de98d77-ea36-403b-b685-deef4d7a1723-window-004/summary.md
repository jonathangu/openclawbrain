# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-570958465e5589f279bb28e850af48f0de0e358122b512d402db3214c7541c3c`
- fixture hash: `sha256-06808b26154de9486de3e390d83e02d5c54e1e0ca160f5f4c88501af04825dc3`
- score hash: `sha256-d0ba31dc630b06648c7a2388171223e2c8cc87937ee184f80b828e224ebd8f89`
- bundle hash: `sha256-190be16ecccad480229ff6fe316bfe67a74eed63277c3dc83ec564b5e9bab9ec`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd27477bdd5733b8ef83edfac9b06aafa0bfaf3753550669b2a8358e4c2d729f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-40396e31c25b123a501ff635ca92b5e7fd334af5adeb8fa287e316147e3e71ef |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3586f390f36fe3317b8459acbc809c94636b4dedff270df5aa7077b2bccf8441 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5344e9dc497c28db4f4c2ef6b186f06e62a004da34bdfbf51c558938b79f6bb4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-991a1130 | sha256-34865af611769fab88278d317fdab4cd8d643b58da5b08d4d126edc31b6447a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-991a1130 | sha256-44d3f6aa5da6366be1dbfb1f2df95df377b9e1ccd7b6db68ec9d2f962d79a932 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b2341399 | sha256-f71f5aa0fbf92e36ddab1bd07f5bcdd4c9bef09aa38e09fb27d838facb427872 |
