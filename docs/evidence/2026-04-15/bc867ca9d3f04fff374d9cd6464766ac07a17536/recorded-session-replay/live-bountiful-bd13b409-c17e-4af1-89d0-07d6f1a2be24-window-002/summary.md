# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84f2bc85f266bfd29b006c2dc6ef93b9e165e29f55fc2011158e761c60436082`
- fixture hash: `sha256-810d1c824da649ceb29c027b026aba0238fc9fe617410c118b1a8cf769536f9f`
- score hash: `sha256-013928a836e522dbe44ff7a1c4d2cb12bd5baf91c7d460ab1a37d6cf4cab3229`
- bundle hash: `sha256-c05b333dc511967c799bd5e9caeca6f5d1944197af125c456792a9347e760c80`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1065d3831f028776c526f039887471c1da0a0d15099e1e578d4ef85bdda6d0d2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a278629a2a6cb3a4af22752ddde37bf5375deb71fe531917a58fcc9c804a3707 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d1d97edda6d5534c5073e26dee753a3f1ebd9de603178306e9082653db9508b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-856ee717050f9b80269b040e552f0e8bc7405e33f87f46a1098acf5537da07e4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7950411d | sha256-60ac17e80ec4d01da9395db740148edb84d1e9fc02a4aee729d39152587018b3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7950411d | sha256-1fe978e7c3c611a13d309abcc3ed6ec2506a62dedce58c9a32dc06c6dd85075a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7950411d | sha256-60ac17e80ec4d01da9395db740148edb84d1e9fc02a4aee729d39152587018b3 |
