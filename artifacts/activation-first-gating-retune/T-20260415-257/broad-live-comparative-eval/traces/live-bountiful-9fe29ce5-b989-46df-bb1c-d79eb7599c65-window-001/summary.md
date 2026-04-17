# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3249aba74ff3b68a3a52303cdd5411f6f55111b4c4f3feb276bc9f491c4a0dfc`
- fixture hash: `sha256-ae9594a971d6ccf182aa1cfc577566bae527c792a4eca57afc1a5a898e741bd0`
- score hash: `sha256-9a3f178192723776c531247b6ee15f4027b820845ea8129b07e354734d689a41`
- bundle hash: `sha256-d77f19880ce8eafa18f643a74bcfd4801d0b00a9dad23ca727a41ff3b0c67a44`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cfb1d94a577129d4d3443f4e0e588167e5df8247c7459669699a79d5c108e8cf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-58cd6dcc2bed2973f337171b8f5f01a7b2377b05306c9f5827cf669d07f67b99 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a9d735950e13bd5ea7dcf890468e9f24ab2039f1452675835c2efc00db46e327 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-14379ceeebc7a767b976b65777e1b2cab302de730cc1ce3406ace6fefd372525 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d06e7f84 | sha256-218cc5abbcf13c0df2c77d7ae19b989679dae53727871c865848a06a15e252ec |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d06e7f84 | sha256-218cc5abbcf13c0df2c77d7ae19b989679dae53727871c865848a06a15e252ec |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8ac5398b | sha256-400e1a3c17badaa0ea50d7bcfbb2c1c89c32cf69074417ecf98703842731de12 |
