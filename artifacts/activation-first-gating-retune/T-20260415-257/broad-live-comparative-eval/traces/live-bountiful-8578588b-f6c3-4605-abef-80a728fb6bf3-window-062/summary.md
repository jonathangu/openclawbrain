# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-95a304cc4e6fceac322fff58f14559b4669ed6895b2d6fd1036b4ee05824dbb8`
- fixture hash: `sha256-c3ee8c1fbe9d9a70f4d964351557d76b48d009945b0ecd2ef42662d9e85f4aa1`
- score hash: `sha256-cad5a04e9e7fa62ba3eb1cbc570e18383bc1e1361095846c388e2cf0c4a44a97`
- bundle hash: `sha256-fad4d6d9221c7d718ee4d719a45481d094bad679b6e94055e5d8d807969b8eba`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec7ec8f6799a0ee01f6b4130aafafed76ac2a835827ba3dac19bfdac983b407 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a27f675b34eb0409b9a70e8fd88c71431c5124ec04405efd57d5cb23fbff99e8 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-85ae5c968c0f84919be04ee6c0bf4515a1fdb7b4c87dc9ac8c4e449e1b83c586 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6cc2e6ab1e8dd686f9f137ea2ea64560db4aa21ca9dbaf0abc6d0eb686855709 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-8a220716 | sha256-a7bc63ed2c59c97a3b07da5ef6f17ad583605c588613bb635676be3df9ff431d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-8a220716 | sha256-4079f9b0b9170e6e3e20efae66566d05cb28f6f6e62086c9a6bde9e622fe3954 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8a220716 | sha256-79c940551b87eef9662644dda18f5f6d6473a47a892cb443edf7e31ff0e7ecce |
