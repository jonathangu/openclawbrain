# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3249aba74ff3b68a3a52303cdd5411f6f55111b4c4f3feb276bc9f491c4a0dfc`
- fixture hash: `sha256-ae9594a971d6ccf182aa1cfc577566bae527c792a4eca57afc1a5a898e741bd0`
- score hash: `sha256-afc548d1ff2b18e2064e1b775381b4ee3bba6cacfa72161c622370676ca1d58b`
- bundle hash: `sha256-75940a41f32b1b027236a7a425c0e0b6ae0c336c5e2a016848f69bf36d83a045`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cfb1d94a577129d4d3443f4e0e588167e5df8247c7459669699a79d5c108e8cf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-58cd6dcc2bed2973f337171b8f5f01a7b2377b05306c9f5827cf669d07f67b99 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a9d735950e13bd5ea7dcf890468e9f24ab2039f1452675835c2efc00db46e327 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6b7890ea1df4ab6daaa03e7c792c185473206f12f63435780dc414a4b5fe8ef2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d06e7f84 | sha256-218cc5abbcf13c0df2c77d7ae19b989679dae53727871c865848a06a15e252ec |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d06e7f84 | sha256-218cc5abbcf13c0df2c77d7ae19b989679dae53727871c865848a06a15e252ec |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8ac5398b | sha256-78debedf9e90d84795bc37ad1731c2660612e0a0ae1422a0a8f63eec7ef3c5f9 |
