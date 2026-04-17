# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06c88cbd7b40857f6269dd03d5e04022f7a27c8c5e2a225bc79b2768cb90fdfd`
- fixture hash: `sha256-6d62bb5ab6456b9eec73e20f3d1a35ffc14e9452a4f4442f3b56ae134f63d27e`
- score hash: `sha256-57d43e608f08e1edc7cff6a4d1568c54a9ccc914a27acabc68df1bfd4dfb06b8`
- bundle hash: `sha256-a460963b5c1732313b4dbfd00b0052060c0f12b1accae1145d9f818f8a99cea3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd163af8984e87c72885a17249c9a84973c54f74e5363d963d16ae86c9b4e43 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a99b860fbddc981d14d3e234f33758cac494c9bab810e937f5a413038256d2c0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-daa7844d58550f18931e6f991426fad8859ec816b9a93bf0eab4edaeafcd9477 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a8214ffa0084b66fd4fada1612659a18754a9ff4dc89f3977d5b7ba57d0921c3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5c4d893b | sha256-1eb4ed31197325801862f6b3c1d4fede00fe6659cd7c71a97ed818e4af6fe3b9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5c4d893b | sha256-2f4306437cf313d83d5361304c7f4713eeed07088fe51e33767610d1a9eecc11 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-78bf8eda | sha256-5e6e638c12373da121118a5b48d30780503d66d0cc59660a848ad5ddc68f2faf |
