# Recorded Session Replay Proof Bundle

- trace id: `live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ab9e7c92fecacf60147f27b8c27fc049edab247767d390d6cfd938c3433d0a10`
- fixture hash: `sha256-924c4ed1186166bf0f6b342b0967c241c06f6d79d3f88b2eb96f947a6b1061b7`
- score hash: `sha256-59d479d66980b23495b6947a35e779656444f7cd1e2cb2256e1ff974ed26b244`
- bundle hash: `sha256-8c35ac66aa54264f921282a2043f838513846426787f9136b01afdcb78dddfba`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af4d866d80b0149b53a0921726b1499466ad574098653e689171c2b2c56dcca1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5cf9dccf621e0081e7d6bf7f9b7ee4fc8b0edd3df537b5843720553b2fedd031 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ccc48b2b6d61597e3b29096852e27f80703c00b9670a4dac0061ade2a0e7849d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c8be0ca00acf45a1c6e4adbe485e29206d3390941147395ed47fa3035fb4d06b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fe7531c2 | sha256-fdeef86dedc8054f80ced5f14039a7b9441f999d1f43f48e658f876e38bc68b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fe7531c2 | sha256-4f48fd253f1fdbb61c707152f32c4ed09f901d52df2335686432810740a06e54 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ef8b5a53 | sha256-42dcfaa759364701e96f40487dd1aa536ad78de5320eb90cc7f034fac3e87f17 |
