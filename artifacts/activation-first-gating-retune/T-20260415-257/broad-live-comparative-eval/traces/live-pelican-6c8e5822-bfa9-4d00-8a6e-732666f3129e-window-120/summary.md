# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67782e30fe5f9982125f26c2ecd77317f6b86c34b8443a476ff968e4172fc9ad`
- fixture hash: `sha256-3275c723fd5e55770c99a0a3826bd67e0749405b630c9523de493fe0719c674f`
- score hash: `sha256-0c7dc13f39568368d0f7188a94c5cb5e8860a4bdf2aa70dbbef4b7c8b6fab64c`
- bundle hash: `sha256-a2b80be3596d1319162727eed2f6052d742c6f88bb0a59321f4ca5012b4ee11f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6faa741f27c297696cddf75c51e07e62f9d376795b5d33f012fd6c625e199a2d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff48e0c6871419bcf5120178e6e47d38331511ba6eeae6cb88a1f2769d4498f4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72a54676ab84c88e9aaa7ac6aaff3404182961ed8da4e39b2504b3729245f9d1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-88816731bbac202d99f780a1c503f659b83f8307ea669c7512d722705a668a5f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-994b2999 | sha256-c47a52d02d5536695ef95a31887b2f1b315262a662b7203fae0f2178917d7636 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-994b2999 | sha256-bdff8f5fb9f14b470a020643c864f0bc2d8fb132041d76c64f031d1fced8db40 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7e7a7480 | sha256-9b06eacd7b0f3a2c330a4ddd5dbd605b12bfedf4ea5944458e535429a433035c |
