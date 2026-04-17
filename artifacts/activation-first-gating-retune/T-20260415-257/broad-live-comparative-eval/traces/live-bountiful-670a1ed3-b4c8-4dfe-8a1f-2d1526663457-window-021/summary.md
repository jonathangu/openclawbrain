# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-512c430e649faf76044870db1348b61987384f6c4b42eb2624038c368ab6a4bd`
- fixture hash: `sha256-6f2c5641408f7a03798669e19a288492bcf8f6f0b8043e459e2c72b4bc2ef9f6`
- score hash: `sha256-6161320c80d74db1321029380d0778bb5af61f6107871f4584b122b65492bf3d`
- bundle hash: `sha256-a34fd9ef8db13f01b7798d6206da7a51bc2d41d3111748c6d2cb4441e0048e0f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fc8b4083586f10fbbdda0686c1eb4cc964fe1c89c35a3824fb52431cfb03e36 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39aa9041f7704b58123ff4fcc745a23de084f3735a8110414a53927758361e94 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e7a9787ecbc77aa75eea2bdfa1d362fce50fcbefe679c0e45b5f528ab3cba35f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-477ba6caaec62d30464e52e49e48a4e4f9a593ccc3e711c18fe82081b4fe0642 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3b50fbba | sha256-dc539bd27465782b75c067736e8d91dfa065add1895ebcea5fea138253b47f2d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3b50fbba | sha256-60eae39cee0a7b6301f708acc602793299589ad55b3e71354d86857c039661ae |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d7fef7d5 | sha256-4ade950adba7292e749a1dd1b8e8af199d49f2e70db47c6c99ce3053acbdff84 |
