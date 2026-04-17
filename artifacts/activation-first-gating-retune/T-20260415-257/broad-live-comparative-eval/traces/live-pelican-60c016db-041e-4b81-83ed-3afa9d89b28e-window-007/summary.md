# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bfcdb554e6c3bfe187f4c905f92a9b282d7821367cef535897c2815e123fe75d`
- fixture hash: `sha256-3907274214cdd60210f9dcb9d9b0e865d090d5365a59db918b98e4ad4849f4e5`
- score hash: `sha256-8011836b8d2eca16c3b342718e325d67907751a2d786797a013aeed2344d0e64`
- bundle hash: `sha256-897283971381bef069cb023bae4e04cccfa775c821a08cfe280604af29f3d8d6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18ae191771eabc01fba0eef9c0e7f277194aa1ae188e2e94481f667ee00cc41c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-801f97b48ff3bd32cea5fbdb331be88014bb014392fb6e70ff12dee198dbf2fd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f7b0647a0437906a31086e0df0e6b4be1dc463afbeadda6256349f8156b8883 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e38c80c828d1228cd7015885ebf04db9f6a9a79266b5fec197990e11418dee20 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a263f28d | sha256-2ab4d4356812711d933de4bc4334e9d1db482c1063ce79d8348bb7504cb37c12 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a263f28d | sha256-fc83d602339d52748eeba90fe678df3bae04dd08ed8fba7630944d678aa0aa62 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-704b0a16 | sha256-09efba1cd3d413c636b3f77f385828d4226ab4406107a762900bd61d64227d0d |
