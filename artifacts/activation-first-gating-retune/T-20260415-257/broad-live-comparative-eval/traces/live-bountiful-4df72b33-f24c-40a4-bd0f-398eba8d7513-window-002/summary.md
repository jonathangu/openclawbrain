# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-25e681dde9bf99a5066e3fd272c254e137908dd8248f9cd30c28377b5642eb80`
- fixture hash: `sha256-118dd0d43d47e09d3e0fb14557115fffb91ecc9b2c9362bf193950d5af577035`
- score hash: `sha256-6882305878eedf8302df93d76bc22bca283488327f289f4b418525b4ce9066e5`
- bundle hash: `sha256-a5e5ea8213f7603002fb8928ae4fab29fc613158d36b57c9ec99572bd21ca18a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | vector_only | 80 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-053f07f407b9f0886975eb3e4d95aa7c39bed9e8cf96e6716ec7a7f71273ccd2 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-1d972534439bd247053030f7584bd0432bf6a9c5cd8a9b4a73e5f24729e4fb49 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-8099db21fde60420bf81195f36d4c6299e0649122ad0781c5b1eaa2216ed2684 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b5f6b58ded9f7efbc48f1570c5f759c120f446ce026126b2c7f2950856432d31 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-dd46c7c3 | sha256-d8a88ba491272c1fd214a6a9e5309e759817f22d665dacb8d0aa19f410552def |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-dd46c7c3 | sha256-d8a88ba491272c1fd214a6a9e5309e759817f22d665dacb8d0aa19f410552def |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-997443de | sha256-8c085da3bb4727b02acf7708672ab264ca497dd5d06de9348bba77b9c1434d83 |
