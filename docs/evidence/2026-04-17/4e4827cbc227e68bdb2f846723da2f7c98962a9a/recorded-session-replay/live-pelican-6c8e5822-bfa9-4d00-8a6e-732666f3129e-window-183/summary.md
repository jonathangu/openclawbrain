# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183`
- winner mode: `graph_prior_only`
- trace hash: `sha256-203ac39480367005fd42cf7825311a0cbe85dd80f56721c12d00a8ea3f270b1f`
- fixture hash: `sha256-f1b7e7068a4652fbad5d085cdb0c1a635468b0ae89cc507258e65b4da9413c08`
- score hash: `sha256-12df77b65190c3d3f7494602a56a0f2f823edaa59cec8970d2d68cfd3b33de6a`
- bundle hash: `sha256-d4ecaa700218bc1e1a4eaa7e1a80b33873827dd70c5c4d2cb86595ef8bc0750d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-08044093faa209a549cf6cbe79d77a3fd872d3cdde2c86b5886da5044f650477 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-15a671cfff8e13d9457328136195fff42a5517f0949dbe5561f431d24fc83563 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cd605053b1195401aab0ba0b272a046e1d2a7a40bf1b3978c8633bff3cda7b80 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-0c1edefcf8887adc62f617098ecf9800259f05cf9dc05471e68946c632064461 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8a8b19da | sha256-1800a9bc3ebf78b0587f19fd5644463b59d0ae8e2ae3551a8fb76ae48c0a2cfa |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8a8b19da | sha256-c325149898dce53d29e4d37ad144f0572047b7b89e6f4ec84551515a56d1c299 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-f673d9c3 | sha256-d3116d76d0f6bee2bc7079a9029b0f1d89c8651878b7b67e5b705b2c94fe5e8f |
