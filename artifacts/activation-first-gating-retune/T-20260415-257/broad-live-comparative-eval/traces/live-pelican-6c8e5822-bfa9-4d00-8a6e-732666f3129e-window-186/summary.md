# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e334b58e5431d3b20f7572c904faed7d64f26bc6fd3cb1bf1d055e492134e8a8`
- fixture hash: `sha256-8e788213c51f0225abe30e2600382afc50022c57de7f08753d94aa61dd287dae`
- score hash: `sha256-5583ced33f55a799b734ee56eae567b10c5078c198c2ebdfcc4156c0303b5d5e`
- bundle hash: `sha256-ce518cbdeff1358405c039111577a2a27ee26f213f3eb991c003f33f9c051534`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f4621a0c949a3fba62d418ef21dd1d6c65fb58e546b35333db0f8e5c2c8785a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b9144888021990de8ddcb2c4fd48995cf665f8d004c425a3e1a9c3b295afcf2b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-da34af27472e2ba7e9b4085dd82d0f1884af98dd33a2ff12368df8efd3a21a58 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e0e55bc7af366e5c262a11d047d134b984f528f623fc5618eb4fff8800321ee8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a01f6f98 | sha256-f518603a0b69f36f05bf3e963a741b7d05e66564f09ad9c474688bd42bc08532 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a01f6f98 | sha256-b1dadf2a87bd5940e54466882d6fae324e8065606450ff31d225f7766a7638ba |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-cb7891e5 | sha256-eefed332045124ba9c4540d5f439f36da5d4e12833a7cdc53ba66a058b3d49c0 |
