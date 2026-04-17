# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a2590462dc987ced28ec91e593a00f4b408387f6ec40a92d626a6087fcbd75f`
- fixture hash: `sha256-aace8a3fe4087409ebd528569ab1ac34f47ecd7317117709f7ec2907eaa6127c`
- score hash: `sha256-3aa9f5db4ecb70ecc82be89ff623e4206e3bf4f1475b79a8d3ef635de54677e0`
- bundle hash: `sha256-e4f04609fe5ac5fa3b3fb7fe2999e029cee284844966b814e717a79630f9deac`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2d2d9dd5ce486e4334796b2692780e0b5a1aabacd13eeb32d1dca3c57b5e799 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f62260d1a5da1f3a7e85db6ef1c367a9fdfcff0f4178ea9e0d7fb4a3f730c466 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f35a15ff9af7aa3c562081161312730d0efd0ed3fedd08fe928d7b4200b4abd4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-0b815c3634a6a07d4b06bbff6bd60c271e89ed6c302d043210cc1b97af83e549 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b48bffb | sha256-5001779d175d416398eabbd8cdb78faf7b0ac2a835bd6e53e3d1bd31227b08a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b48bffb | sha256-8b387a8f974e450322e4fb9e7431e4f062deed7a9d666554e632bf608c432967 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4ae08042 | sha256-36598fa628476113a1bdede8b82f0b79b363c35a9b4cf5f016adaa4818a16f3b |
