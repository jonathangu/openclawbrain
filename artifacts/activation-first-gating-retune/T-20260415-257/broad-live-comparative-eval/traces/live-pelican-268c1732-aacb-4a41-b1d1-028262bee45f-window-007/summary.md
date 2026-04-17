# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f65f7ec4c1006917225f6f3df2297434078972719c9016d9a4a28c343601c090`
- fixture hash: `sha256-0846d04b26eef0a1a7c06190a5a1fd4f54e0a1ec3fcf3231ae0df203565132b6`
- score hash: `sha256-dc06b2c2ac9842c33a0bfc0d1e2598b3dcb6f662ed472d7e78ea47f72db099ff`
- bundle hash: `sha256-c49f958c1904d4868d90f3ba30ea0909f3af22de3e9dbcbbcbea7616b1e8a3b6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-42f48b48c6c450f0664e256db3a267d908035a318a1c9a74a979a0b9949d1634 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bc40f4432407a93fe6a37c4b65e6d3d21ab6b7633f140aa85ed284ef5cceaf5b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5c5d1d9c90751e0d695beaea35f0fc58834f7bea7ebfaabacaa1961ec9e1cacf |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7dd58a175d516788042e0b1a85c95a2e4001dd29509531883f8752c05fd4d7d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c2fa456b | sha256-632972a6ce93d6b41004dc2262d22a07b3e47615f9f6fe90b0213b0a4f29bdee |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c2fa456b | sha256-0b8c51ad9d8f97fc2ac7e480efce119111f505a47287daafc83ac42c08e85e32 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7b4effd0 | sha256-529a490ac02fb56a8d7a7d4bb34934b1fe86d18f3c9805f5b2c7839ea9c09128 |
