# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3abef879206da47064eddd47e25da3ed69b90db7cd3c4a8ad4966415b7f00bfc`
- fixture hash: `sha256-9918ac1f02e6942937a0c165ef4e1221b4c237d331f00ffb8e89f19fa2868433`
- score hash: `sha256-1fb623f298dffca8bd5e1f83fdab1929d2f8ef9d8101294183b89165fa2545d8`
- bundle hash: `sha256-66f4264a24b52b33f72653acf2ff1601b8aebf194007cfa47d842a3741714a4c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29b14ff43615dac430701017e1a95d84a605d40df7e69393e02bc78849368384 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-40d9c94a15a5831a213c4aba214f86d52fc76e52f0d53b75321b18c779a4520b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4308d47d46e66f9944b1b643bd69b11e6ac0e4afc237a053ab9c2ffe2bd5bb09 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-2f269e611929ff9bcd493f4e06f5c3dd350c03227e9e85bb5886b59dca36ccfa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-0b80e3a3 | sha256-34989b8bbf6609deca84b612f747d9f52d9fa8216ada57aacc1f1bea048be2fc |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-0b80e3a3 | sha256-f0e7bc5881e9f541a5a2b11d638497b51d43ccdc470c78658a624401c0a01474 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-0b80e3a3 | sha256-34989b8bbf6609deca84b612f747d9f52d9fa8216ada57aacc1f1bea048be2fc |
