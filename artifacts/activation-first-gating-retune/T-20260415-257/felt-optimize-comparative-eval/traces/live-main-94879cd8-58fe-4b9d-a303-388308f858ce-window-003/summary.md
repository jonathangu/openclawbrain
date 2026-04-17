# Recorded Session Replay Proof Bundle

- trace id: `live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ab9e7c92fecacf60147f27b8c27fc049edab247767d390d6cfd938c3433d0a10`
- fixture hash: `sha256-924c4ed1186166bf0f6b342b0967c241c06f6d79d3f88b2eb96f947a6b1061b7`
- score hash: `sha256-c93d913e48eaed1ee2bab585e9a957ec57439591d2da72731a0f75a930c8c48f`
- bundle hash: `sha256-253b4a6e5999e94c463a22d66af65a52a749d228a8ff193b55e16336e5b093c6`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-26655331b5db91dc52a3b98b3bd12def0d92bcb0130828e58bfaa08e64d87965 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-517de1e67dd64ba9b48918d5d4a8ff69b455abe0000cfb0be8e8862820adf199 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1c243b9d0ea4d1a02e9988f990ebc109127f3f593a1b57929a25df7f866a3910 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d636aa4a | sha256-60f08ce29a6acb0654dbbc71e47cb06743e7e8753c27e9d561d53950edf6ca85 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d636aa4a | sha256-0862210a491f632ecd4c5efc1b82c22f28ed4e951d5a3a6a5432d63ed74e3521 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c74cd2db | sha256-510a25d2f6eaeb8d74d19cab095e03d87c85316e91914f240fab62b931d1e545 |
