# Recorded Session Replay Proof Bundle

- trace id: `trace-direct-answer-reproduce-eval-command`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5bdf7dae6f318a437c25599d1163e02df616c6f8d8831f793342a828e48d8f56`
- fixture hash: `sha256-9d178dbf6995a5ba8dcb2d1d1ffe4156b462ef915f606d24f8e99715f78f6ceb`
- score hash: `sha256-2c8a0411f11abbfe90ad573ed3040e1dc011e58360a96e86d7295fdce6074196`
- bundle hash: `sha256-e7dda77e05f2b43f9f83773c975693da60ce202e1ef99c05a707f87175f7ee4a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/16
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/4 | 0 | 0 | 2 | 1 | 0 | sha256-a2a7115fd8aa4c7bf4a719e6ce12d1ded35b497046b9e9542ec7ffd21f7790fc |
| vector_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-52e3e2d865a135c0b03c88160184909e3a73802d1699e209039a9b1f8a11f29c |
| graph_prior_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-c25c525cc243aa6bde58517be9145058616370b3036ad965e7a014c2cca79471 |
| learned_route | 2 | 2 | 4/4 | 1 | 1 | 2 | 1 | 0 | sha256-7cac351e084dc01e55c46dc459806058ed85a8ec94fbea21dbefd05ef9432f84 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | reproduce-command-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | reproduce-command-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | reproduce-command-turn-1 | 100 | yes | 2/2 | no | no | pack-bd1c118f | sha256-211a91055b64388940a75bdb55f4d699532e5efb22acb010ae454944731f12a7 |
| vector_only | reproduce-command-turn-2 | 100 | yes | 2/2 | no | no | pack-bd1c118f | sha256-345ebb8bf8aa74ab91c494abcf99cb7a598818261e0ed0ec8445d59252452228 |
| graph_prior_only | reproduce-command-turn-1 | 100 | yes | 2/2 | no | no | pack-bd1c118f | sha256-211a91055b64388940a75bdb55f4d699532e5efb22acb010ae454944731f12a7 |
| graph_prior_only | reproduce-command-turn-2 | 100 | yes | 2/2 | no | no | pack-bd1c118f | sha256-345ebb8bf8aa74ab91c494abcf99cb7a598818261e0ed0ec8445d59252452228 |
| learned_route | reproduce-command-turn-1 | 100 | yes | 2/2 | no | yes | pack-bd1c118f | sha256-211a91055b64388940a75bdb55f4d699532e5efb22acb010ae454944731f12a7 |
| learned_route | reproduce-command-turn-2 | 100 | yes | 2/2 | yes | no | pack-4621b7bf | sha256-99d7cb3e21f3a96016c85da578b2975416028a157fbd7f9cdc5c76d473073e2b |
