# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-694cf444538867e625d49591796eda7824a3f9914c6d50782ffa8d2751091f0e`
- fixture hash: `sha256-cdb2b18e3a901c8928c86a3e5d6789c9de0d594dce56653b0cb654624b8e744f`
- score hash: `sha256-0b45432175655654d10733c3c9da45b9dbc0a7426c47b01b9aa7dfb482747b57`
- bundle hash: `sha256-de503126f0294b2c296ca435ccec1aab18d4afdc88c530e69d34e7a5be364aab`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5406b4db2619fd299a4dff36fb17ece03d149828bde2ae07870bc2e0cc31ba06 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-148f9586a60848d53be4394c7718272d6ee2830e4440e4be8d043e83ac292640 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9995ccd8d682d4ad16e36e718f92bf66416e510c42d0b6b4870472ea13e72339 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-2c9b25fd20298ec70abda421d5019b842c878979440e8f92ee2997eab5a03df3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-163fd401 | sha256-e79588cbaeeae37634f0543de4ccf30b3fc4336d7870e997cb6d71609d778e53 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-163fd401 | sha256-7561e950695dab32c9f8ff46b6e85b79817fdcb8b2844316932c64b212a8c570 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-13fad6dc | sha256-e3b0e2a3d0d63a02dd03374997f89f7247fe5b1f577e45b7558dd77cd8adda17 |
