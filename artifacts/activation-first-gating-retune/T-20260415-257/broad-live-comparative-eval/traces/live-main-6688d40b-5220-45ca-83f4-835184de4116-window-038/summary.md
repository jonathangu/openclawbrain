# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97f203f8cf6e54b5353937fa7c2a9de19dc80e0c9cd1f7a7efe923af5d952db1`
- fixture hash: `sha256-125dd958dc76d20d11fb8d8f175ae0fed91a68b28d91d5171d2943217403837a`
- score hash: `sha256-a51101650f2a7426b1a3efcab37a17b1a33d227adb3dd293be1a7d6f884f7753`
- bundle hash: `sha256-29236086ca7c4a73b47a7741616646f5f1af717bc95c760c5a6e45fb7172130f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-511d0fcc64563746c6c18b192b94f28492b7e276c306dddc6df1e62d381fef89 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-059f801d71d28373445541b9b4fe15109d117c4704ff242e170be6cf0fd8212b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fa86fa8f3669b943e5907ed1e35d5f95660a6fab2052f5be0520f3eb90ee98e0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ca517eff2740ce5ca8ee5be9f93a93a7b9fc57562fd30bd2dd07afe4f52cfe98 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bb71636e | sha256-249381d3601d598479318f346ae6e8975d1f3ca62e1e8c564ef3203d0ae73569 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bb71636e | sha256-589fb60bcaf1f50484aadac4f293d345059853ffd290b040294d3cb091be8e13 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c9ab024f | sha256-96c0b7d31a24b357156c30ddc0f8bfc77d41adbd115f8e366259155739719132 |
