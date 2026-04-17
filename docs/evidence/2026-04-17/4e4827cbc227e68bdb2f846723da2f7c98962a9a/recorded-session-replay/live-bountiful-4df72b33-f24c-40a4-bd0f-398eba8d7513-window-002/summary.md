# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-25e681dde9bf99a5066e3fd272c254e137908dd8248f9cd30c28377b5642eb80`
- fixture hash: `sha256-118dd0d43d47e09d3e0fb14557115fffb91ecc9b2c9362bf193950d5af577035`
- score hash: `sha256-a8f8e13384456cd05e798577a641a52b55a63fe1bf4d57c8a262e8abbda95af1`
- bundle hash: `sha256-448a8ba8ad48748fbb513b688b1d88cf74382dffacedde968f35746b433f3b2b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-053f07f407b9f0886975eb3e4d95aa7c39bed9e8cf96e6716ec7a7f71273ccd2 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-5dd9140c627e0cd0e4f4ff82c4ab9f46865a61b238b4f310d6fb709cb85aba11 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-3cbff4113d519dd4461d290e95b8812d37ca926bb43c7b1524ed66e97c66d9af |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-86dfeddc1c1d00ebfa027538173fba09eb78f9cbcae7ae31682a445d57619690 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-102a3d92 | sha256-9e39eab75db0e727470b22306c669fceb12f10ea15f84b2db35080b2ff3db252 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-102a3d92 | sha256-9e39eab75db0e727470b22306c669fceb12f10ea15f84b2db35080b2ff3db252 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-cc57b9ad | sha256-98cef9ea636ffe1afa0ee93c5e61808c68dc97fa7c4c067b7c64bdf3b185a8a7 |
