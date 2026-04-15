# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055`
- winner mode: `graph_prior_only`
- trace hash: `sha256-069c659e483d79099c9522902169a0e3c008a2a3a1a608f281e5842abe60c793`
- fixture hash: `sha256-2053a334b00cb8986b08e94b050daa206b7253e27c1f42496d3a7ffe4c19e5d6`
- score hash: `sha256-7f9d87d994310d7b7242772d229fed9270cda5dd195ae78fd0ccd447fb2c731d`
- bundle hash: `sha256-b8c3230cde2330cd9584d302bbb1ff488ce767e737c085c150915d0cf8ccd0d1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c7d0a9be6adcdff721e255d979e1bead77026cd16f5da5ab306eca424cee158d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-05e080c94885cddce5eda805ffdf54c1115e84730ac6262de69696a219a931ba |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0d2b6ae6f266726bb37e99611760633ab5118838d46fd18d5feab4f4265f9102 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8beba5cd5c2df47081c227e9f8fe834cdf395a2312b6a4c5379159f5670f6a8a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1bbc2dde | sha256-0ea85696ece3395111765cd6ec47220752b3f523bad933dc35c52ab3c8160a6e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1bbc2dde | sha256-4a3a6a6075dc8618054ef32385749ad954c2a1615cbc6ed6d9f5a19202c7d544 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1bbc2dde | sha256-0ea85696ece3395111765cd6ec47220752b3f523bad933dc35c52ab3c8160a6e |
