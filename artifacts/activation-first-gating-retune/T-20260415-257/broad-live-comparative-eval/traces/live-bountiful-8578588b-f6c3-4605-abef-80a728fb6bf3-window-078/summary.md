# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-09e16d33e75c4cf1bb693ae2b746367b7c597d6c1f3807bc023d0caaa70d4704`
- fixture hash: `sha256-5891ea95069e0741e64b63dbc158c08dbbd916c4d462cb19e66a9822069e3b77`
- score hash: `sha256-999868909f3eb69f584de78a4e090a242b25a26f8c22df0268722178497a729e`
- bundle hash: `sha256-ee52e992342ed981abf9251fba11ac27ac5b5075ea6cfaa2335a0cbc8b7a8f35`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c24d7a3f72e63cefcedc0db743257682eea37c873db139e3264a8ee79b5194f0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8772125a802051d46282385d839d0cce476ef647ea8e684e68b5154679d7940a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-958a82bb918c61490829c5b6bf8fab7bcf1871040c1ff3924117ccfbcdc0538b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b8d4805b42f3e4cc41693c5edde1b620111bac155d22d2d2da363deb8f499147 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4fb066e7 | sha256-44b39f8687ca2fc2534bc20a3e421ad4bd7ad785e121443e0eace6023e5e1b19 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4fb066e7 | sha256-53b487cb91f4f352402936907df74478272046903c254113b364cdac470b1b22 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5531588c | sha256-83bd5e72f8887af32d54298be16880acb5b2ad3578e2add9342c465798077a18 |
