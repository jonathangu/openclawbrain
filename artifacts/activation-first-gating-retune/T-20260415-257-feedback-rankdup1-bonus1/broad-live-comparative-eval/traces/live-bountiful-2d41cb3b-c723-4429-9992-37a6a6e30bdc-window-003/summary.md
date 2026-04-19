# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-580d4f8ccc5672e0994a6e33aa91736865fe4849bdf6f4307be6ada1929aaaa3`
- fixture hash: `sha256-f2131456430264646f8b93eefc85baa48a48ed730efbc3d47ac8e04c07a9e06b`
- score hash: `sha256-242549a361651d33064c71511f9809573736c17abf2c271ba3c68e69c47c79bd`
- bundle hash: `sha256-2bed2566e392bd2e9c485925af49cdcb4a9aec5f6c88f3c48b11254f04e90cb0`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f4cf4becbb2f48d6942aa18f5479385f6a9e60c52e2114c0c244447f1e9cb2ba |
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-a2cfdb3b38702b86698f1606624f0810c39b4d9e88682f7baebfd874690c00a6 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-35bbbabb2f1a5bad45728092dd145566cd0e0958a86c3085907cbbebfd3b3c95 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-84ac9683c8ee0d1ec5b0774b14bf6833747560b4ef4c1f464922c4cc56e9c8ad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-ca601306 | sha256-52f9dd1251799fe6fc23399e8f344babe4cb86111dfc59923b13cab1f42058a6 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-ca601306 | sha256-d00a3720b4d4f5bf7a95824b162660ec4f7c2564a02a4b32a8fc1f85d0a32037 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-ca601306 | sha256-52f9dd1251799fe6fc23399e8f344babe4cb86111dfc59923b13cab1f42058a6 |
