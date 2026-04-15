# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-32e0b4ec2c1ecbf5a44b66dab5340f30730d05ccd8fc6dea8e459b03d93bb729`
- fixture hash: `sha256-cd231e74dab2c7ac691e39a4ea475c769c350fe4115dc674162e2af0c0f3148d`
- score hash: `sha256-8b0d752d0fcf04ea140dbd6b9bf4888c36736f997fe29a9873bf28f5835a6002`
- bundle hash: `sha256-ff93bbe51d873a3310060a00a6d99c72e57fa15e2cd9811edbc23bca43aa143a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-eafd155fdfa2fbd8e1c5739855382bb4aee55ae760f037b37bc2cd66c8f2b4cb |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-2819bc68e9c214e2266c5cc9ca721f825a621dc28ce1df17caa0554a7f368c9b |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-4bf7fe83a6cdc3f8fee35933a14a9b2207fa3812b0a28a53e147514ef4b90598 |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-709818035ea6a6d7d4ce28f16450fb23ad900bbadcb44aaf5b81c25619a7183d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-54bea220 | sha256-ccad1800bc0e4cc36df62dbb4afe6a8811c194fe2250fb878ea87165a09f58d5 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-54bea220 | sha256-9a5886f8081c1ba64b747d9ced17d7dc12a19e2438d7104827be76317ba4b610 |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-54bea220 | sha256-ccad1800bc0e4cc36df62dbb4afe6a8811c194fe2250fb878ea87165a09f58d5 |
