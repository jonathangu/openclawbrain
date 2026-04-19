# Recorded Session Replay Proof Bundle

- trace id: `live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e321442dc8033dd76db95133894d776ec05ebee5a5a98eec612f6b420b907658`
- fixture hash: `sha256-742118fbdeeb061b08c45664c524844d158f1b6be0af589fa277c4ab60f660e2`
- score hash: `sha256-aa6af172fe0cf504aff8fea078e19f37b6dbbaa922b83ce0f075e1b301391957`
- bundle hash: `sha256-93c2e94df6fb3b4174829a46a906ebef7bbbcf74e578dfdbe6adb593e1a93e51`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-05b0912208f70d1fd8d2baa8f914bf08175b3f38b8f85e68cab4f50d835557ec |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-ba53f0f207a31014dc08d7fcf941b30d25c5dccb122bfe1ea21c264cdb747cf3 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-6c520a02fa113af3671983a765ef6c1ffcc78bcb633409e67bb7cefa916dff5a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f4872a21442058f87ba7aa63762e2e02f7c15cc74f934ac20affda8492693fd1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-6694bdf5 | sha256-a8a6056d1d9105e56bc7afc73b11719aea1210890f1c709828cb62cf5df3477a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-6694bdf5 | sha256-a8a6056d1d9105e56bc7afc73b11719aea1210890f1c709828cb62cf5df3477a |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6694bdf5 | sha256-06b6d27788bff8a1db4bb4873972f27515b5f66a60b6d572d89a2c107b5daabc |
