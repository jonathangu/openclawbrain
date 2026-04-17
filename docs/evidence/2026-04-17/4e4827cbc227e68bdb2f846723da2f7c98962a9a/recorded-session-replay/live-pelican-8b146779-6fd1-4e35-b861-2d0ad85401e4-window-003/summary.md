# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9ea17f3aebef4af75f0a93c521d6e776070c7076063ed113dca780cff0b9684`
- fixture hash: `sha256-e43f09daa5c7f1f8012274d4f09baa27758aaa51c3e914baa4ee6b5329b895af`
- score hash: `sha256-062d6a8e25b55c1e956ea391933fe7e2726f5e39a4b1d66b725fe58dad018875`
- bundle hash: `sha256-318062b90a155cf2ba883fcafebad919cdfb15245f6b8e0b6bebe6811f8abd43`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-174fe02cb9d576a687ddb560851b02ab0e12cb6737fa301408229ad552fa41d4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-37372be807e77d8af7f6331f9e65ad947738759f307cf9929ce81571e5621c77 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2e4bb5ff926c264a85df81975229f943da2b13771b3224ef8638c7552745339 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7a62ceed7f31f53b2372643ab96d533c536977c1c8ab37f3c7384bc33e112518 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7f7576d0 | sha256-6e55a2909ab907819082d14dc5fd583fd034b2c22fa695810e388f71186efafe |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7f7576d0 | sha256-053b4ec82c6cf50598ff33d080a548df19055191301cb77928a85f76f9cec101 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b7d2084b | sha256-691f8c70ae5a701c2a5daddeb82a4a9d9c25a2d75c8da9ade16443bf12693c48 |
