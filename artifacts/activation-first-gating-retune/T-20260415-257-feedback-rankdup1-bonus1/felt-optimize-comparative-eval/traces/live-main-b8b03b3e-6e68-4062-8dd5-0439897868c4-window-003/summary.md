# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a2590462dc987ced28ec91e593a00f4b408387f6ec40a92d626a6087fcbd75f`
- fixture hash: `sha256-aace8a3fe4087409ebd528569ab1ac34f47ecd7317117709f7ec2907eaa6127c`
- score hash: `sha256-eeab3ad2f2d248637196dbd5f2a9277e91455bb7f53467f707b687ca69fab6db`
- bundle hash: `sha256-c22a443ad92de22734e718ac9ba611e7cd60f51e4e54d262feb0759d644fd622`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-ae69027ee69deaf246191c43e9a6add1e5a6de947275f50d4f76e4958ddf6ddc |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-98d4fb2bfafa6e334d99bd5be55464c136a1dc8da096b0822e65af1b65124b6a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-33c531d811efa8ea6836b57ac384134d4d622deba824ff36a077919599fb514b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-392cf5ea | sha256-96f90cebbe1351c91de4d7a75b43fb002f94d9a7d3513f1d461d162bec6d4671 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-392cf5ea | sha256-91d792c86475fdc65bc1726e8b92b436307cfc3c8d69734955ffcb1516a70e0c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-392cf5ea | sha256-cb86ae0a2f8423fabe63ee120205462ef0b8f23eb19ab62cb1b500cd86f13e1d |
