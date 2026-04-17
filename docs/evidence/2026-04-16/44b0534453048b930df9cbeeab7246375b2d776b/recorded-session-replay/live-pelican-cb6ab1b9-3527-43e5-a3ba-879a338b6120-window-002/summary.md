# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4ecb7ad01ebf51dd3aeb7754e784e70eea9f4067a9392ad81778aff88de83b03`
- fixture hash: `sha256-ad7aadbe694390cc07af980435b05bd2086d5294c79bda5f4f75ff348a4a3b75`
- score hash: `sha256-bfe4f55e0aa2282033ad62647a2cf8f178c515b700ed31d1632dedb6c00ae35c`
- bundle hash: `sha256-4afa5edc3bdbe320bf11345d705009b0a20c255d7336cc36b3d92e6e6d4cf1f2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8a8a0e6d7dd1a7545143681fd0202299acfcab2ad5ce85ed5e5cddd516c7f67 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c228f6883caac6feaf70e577fd13fa6bfc97e076975fb5c9c98b988deb3af34b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-57fb824a00733aa6a420f5253c9c8cff6c4ab7824fb316b3e5d53c920710d088 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a89a4ab0a12b03cf2c2021c4ef4fd4f22e47ceb3e5a9d073c0e303c69976a4d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-10a2ee5b | sha256-9ca62b191e2cde21aa4b516c3cf6bfa937196ce17df543b3dea507093eee5d53 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-10a2ee5b | sha256-23cd1ce82d9e6a235f1e7fea648d61b43ad6bd1a008c004542015ecf00a2fb2b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b0950d8e | sha256-4f7da019ed946f834d2742413dea6cb19f3e3847303a2035035dc8b3ccc28ee7 |
