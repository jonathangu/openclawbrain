# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d68f21a3db07e083abd55cb8f30309dffa35aea63874e95510f19d0d69cb1ce`
- fixture hash: `sha256-370af296b8752ce6655fe59921b05e957209333f8adae37b056699cf10a9af35`
- score hash: `sha256-f0d76651d8055a5ef8d14972e004a0a8f5a7a64d9c31c0b640afaf9b4f8c0eaf`
- bundle hash: `sha256-1d8ebd2e75cb4bc8a87d1607888e8df371f712ea395e6d3a7253ec5176c838ff`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13ac363ce7285d3640914d39071894fd6c80687f14f6807f8531ccb47249088 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6337cd49eed5198c2f3d65542dac43d6c307ce1b8b728adb2512500ed9d30a2d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9245f28e03893eb3b92872f382b1f4237906c0544d58f257869070529c34e00 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-77183dab0a1fb3fb333b94a0207b4a07388b817b5f78d2bd7543715683471a83 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-184f0c2c | sha256-64c3338f11f6af75f51b7f416ed4dfae41af4443c8b2048fc46d8668cd369816 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-184f0c2c | sha256-e5a4922036a396630eb89c924e21ffc63fbc40f554918fdbb0c344a523e0878e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-184f0c2c | sha256-64c3338f11f6af75f51b7f416ed4dfae41af4443c8b2048fc46d8668cd369816 |
