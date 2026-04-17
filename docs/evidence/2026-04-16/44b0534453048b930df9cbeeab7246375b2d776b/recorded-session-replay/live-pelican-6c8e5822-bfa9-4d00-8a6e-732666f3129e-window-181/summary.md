# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50cbfb4de0d67c0910ccd1f15acc9132454b767d6a9ef6092fa51c701d086751`
- fixture hash: `sha256-ed982aae33c06dfcffb629c09975a63d396b69570ab9ad349366a4a66aa757f2`
- score hash: `sha256-fac30bf4a39daaa43761f7405fd67a5a331f7e948ffe5b1aea02c3401abcc267`
- bundle hash: `sha256-8faa22427fe63e196ae45c2241e856e3e4425a03dd123848650df50c6087f0a6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ce8acab11edc00f581b930ddd46ccaeed311548b8f75f0398d0e21fa5078567 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-460f5bce76d505ffcec7cad7c48ff5ebbf5bacb3af241a450701205649e50154 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-c196ec786b5757308be69f133a75499661dfcb33ce0aece37f5f7757770a5280 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-06fc7b87334ff0f5387f251cabba2dcc97641e315724a7a13b33b459dd792f14 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-914901ea | sha256-597982989924248a843580e36b5a88a0000b3206b903c8a3350a0c778ad96f80 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-914901ea | sha256-20216a242299a0c59f8f775b75c071db6d49075b3df3619501b0e853194cee43 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-e3022297 | sha256-fdb2aa2513fa2fa2e0358681143bff8b66f52c5af1cee721570d41f58c4ae033 |
