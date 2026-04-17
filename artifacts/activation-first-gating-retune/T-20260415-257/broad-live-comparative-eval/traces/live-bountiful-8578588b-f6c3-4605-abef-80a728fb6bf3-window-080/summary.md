# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7deacd5889e0ad224da79a8b0dc2045294d6fcfe7af40c942dc92f39f797429`
- fixture hash: `sha256-c533c8f4dffd9730ef38a51c383960cee8574f83f672136e59013a1d92400c07`
- score hash: `sha256-9fbe0424afc57738de6cafe6b6edd94fdffcd4da13d1d3ea566c768ac7c8c8db`
- bundle hash: `sha256-afd5ebf3e131a612caee155955a6fc45bdfc18a08fd9413d25789e15cc14e934`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9bb519c15cd685af618227f7aaa909dbc79bba57d56f1160ff54bd9389dfd5a8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88be42f580b1b04b53bcf6fb6a39130e1b5338b1d14f521269f9953630649db7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a2eaf8c32f2f9ce4ddd9d80ba5ae331904a3c697c9f60e23cd0a34a1da944709 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-44f40255052f5a7a2b644bd7716b1fbc64db511a394d7796d68cba69a00fd3a3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3a736383 | sha256-7990b374ec70f8ebf502b52f7809f37c21972422e5b32bd61b38a4067182183e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3a736383 | sha256-63e72185af54e162f1a2c020053065543760736384dfd2b84eabf1a8889851a4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-16a7c022 | sha256-d56f51eb8ce37ba1dc15f10af57a07da7e5308af2cb2f5d6e859ec49e96fe128 |
