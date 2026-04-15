# Recorded Session Replay Proof Bundle

- trace id: `live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-77084407bbc4bca3a65c6d02bad38ef538707ae11378fb75424e9db3f47a8cc3`
- fixture hash: `sha256-1eb4b9074aa4e35ff1cdce5f3e7563b07cf55a4f769a5cba98dce236ab9065a7`
- score hash: `sha256-a17e79605c50771a49d839bf2fd4cb0c90fecd2ea5e0a5dc16b272a16dbb49d4`
- bundle hash: `sha256-55b5bfefe124534b7925bdb2b3837491afc7c676bcabc143c7f1bb10739bc7a9`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0b5c306def10c6fde2517bdfa9ba5d43df83079e99c7ecff7a5190a715aeaea1 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2e778860d1255204c5b69ba5a2c614d16daec924e0985229624a8df2f03e36df |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ceb81c92a461161ccfacaa191555a1222dc8ae6738b2809ae8ed7a69683e8f12 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-50524aa9c17b3bf8f4e60ae8de09d4c982e1d786a11feb88048748fc6cb6a161 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-95698558 | sha256-8d1661bb88951946205163ddbb56b6f5bbe6a8594900a510f780dd2b0d86ecae |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-95698558 | sha256-b3c0ab18da1a134eb4409f80bba6fcfbe52d210b5c0fe02af3186204fc9637a5 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-95698558 | sha256-8d1661bb88951946205163ddbb56b6f5bbe6a8594900a510f780dd2b0d86ecae |
