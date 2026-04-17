# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f8ed2037bc2ff0feca7432af422fbb2b58a869f3969fbcd41ad42699329bf723`
- fixture hash: `sha256-b54fc3ca6fe17a912f89c0806fd3df709e1f6f80264d7323adb898abecd00677`
- score hash: `sha256-c998758389c84b8a831758c6f421a255963947055115ae134c6665808802dfa0`
- bundle hash: `sha256-1a833171a0b88c2e8c08c4dd23cb95674e5efa9acb41ba54a5209ee963c56dd1`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-34738adcfd10c341d46efa61990a3844e1795a8fb18b5ebdc9694342a06a5142 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e5d9a29d3e9b51c2876e470688329237091307f8a97ed9573116375db227f75a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-64cd83e1f6ec638c1902e371b2a4538a0a44655d2d376b90a30be573b409acfc |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-11bd1a8b77906696817408e54baebadf1dd62cceb962853ad5ecc73557cc7814 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7eb1879c | sha256-485d986d9a3ac910ec1327e9e50c024dd1ed8a5f3eb3f18fb2338df13b1ff0df |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7eb1879c | sha256-79169310d1c9fbbba282ee8086b4385f307eff3876b51c3ef5c3364b0440661c |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-79a0d6ad | sha256-5a619977f85ab68f4e5cbb53083c86896736573586fbfdf0fe8b61a86ee01c86 |
