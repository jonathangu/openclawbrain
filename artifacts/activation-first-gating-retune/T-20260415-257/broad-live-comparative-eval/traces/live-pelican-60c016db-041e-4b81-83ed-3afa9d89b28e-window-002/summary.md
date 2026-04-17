# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cfee0e7c7d7a332b7f5a4f24cbf19b04a7b31698d12d254a92d62753684d371`
- fixture hash: `sha256-539cc58588045f4d44638a17795295875c8ed45ffa9d4d266b2c19df9a95dd7f`
- score hash: `sha256-732a8e7512a09b9a540dac969e95ddee0f8056c13d8b0e3b5adef9eaaab1f4bd`
- bundle hash: `sha256-10e80319a042434b83dbe45831318021bfeafcfa932e0614adce721462a912b7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32ce668e6813134ccd828363a96ac1b89f56519737480b00962b4a14175506 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b0d0916ef53ec6e25f9620b30ff35a297271dbe63ae663ab618da7b7dae64a2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9bcc4ed1f972cda295df6df741e2d2567850dfe6b591797ee88ae5eb82118f91 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d76d6fab64cb6b57736950df694943cc72c50afe03f26a23179f14e65c5fda1e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ff25ac36 | sha256-b44fd7b55f5b1fcd5590802e8cb4c70b08a645a8f2408d0859aba3e56c69dab6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ff25ac36 | sha256-e023eb76fb7e33365b7846299c5cc3669104710422012f86b6b55d6e49b32972 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a8705811 | sha256-bbaa47cd5f9f175bd760a52c50169dcd9ffdb5c129f2278bd13ff2519b8ba6ab |
