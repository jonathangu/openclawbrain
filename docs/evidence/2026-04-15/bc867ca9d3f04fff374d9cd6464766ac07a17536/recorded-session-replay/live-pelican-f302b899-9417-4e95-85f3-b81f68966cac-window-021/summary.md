# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cacf2324859afd8e6f3cd4cc1393b48174ec7965442a67bc34f8b6260b72a625`
- fixture hash: `sha256-ca2cd496b9308f9d13fcff6478fd7a04f824cb026dc43bd11af171fcc1a89539`
- score hash: `sha256-8ddff878d74881b11604df7306ac6e281363a0df9cd8268c40f978f91f09d3ff`
- bundle hash: `sha256-4f8643f581700d31917deea46657ced06ad46416b28c4e15b32440a658c25cf0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-591cbecfe0bbc6c84d3223d049bac9d2eb96d473137d7ef277a661d0bb2ceee3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-822de1d2ed4df6d6b615d30c1f934b050e0e03eddb09f88e7afd02165feec7d4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97d2d231d8df8814d49b7614b5df9f93e2480cb4c1e1465dfed030bc6e5f4ab1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-05b4f97c0bd6e80fe1c511ec193c51ec2ddd3e605f6ba0b9977fbe609f87ae6f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-af69a002 | sha256-a854c6b7bec9fe1b2fcceea4de70105892968ae49fa023f65ebda43ae4c67c18 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-af69a002 | sha256-280510bad816de464528890739a02fbc17617dda0ffe3644738816951ff979e2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-af69a002 | sha256-a854c6b7bec9fe1b2fcceea4de70105892968ae49fa023f65ebda43ae4c67c18 |
