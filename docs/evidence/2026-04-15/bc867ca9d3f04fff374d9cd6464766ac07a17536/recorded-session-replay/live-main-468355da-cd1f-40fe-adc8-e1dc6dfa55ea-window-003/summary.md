# Recorded Session Replay Proof Bundle

- trace id: `live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e484b4badd2d1a3a3d24ab18ada126ae37897ad6b6cb5ebb205f801adf4b59af`
- fixture hash: `sha256-7081875ca4f0fc3a1b3a1a20287fd5ff9fc1f2b16a465a1e2418cb78ad0e289e`
- score hash: `sha256-8ef66c6afe9326ddb971059e002a8b44e0f56b5a23276af2c5637c0c18141e4a`
- bundle hash: `sha256-2303ff3d2fcbe9d5f8d22faea3f1fd673caae15255c8e48c0918769544284598`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2ac0d8d192c8a52c6289c0c993dfe551953686d8e0c4d297909e405aea43e25 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-86fa28ab7299e3543f06abd7540c23b7e431763fedd488ead557e341533b3b4b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8dd688ebed22e548a6e1615f036328e4cfa4ae11a29342ed0d4657812fb043c7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b4145b154964c9ac0fad848ebce5008836c06c76aa04fe51d8e4a0eeef298ee3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-15a5970b | sha256-6221ab2fef67f12d0afbd9f73897a071ecb63265b218bc739bc5a6bc279dc60c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-15a5970b | sha256-6221ab2fef67f12d0afbd9f73897a071ecb63265b218bc739bc5a6bc279dc60c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-15a5970b | sha256-6221ab2fef67f12d0afbd9f73897a071ecb63265b218bc739bc5a6bc279dc60c |
