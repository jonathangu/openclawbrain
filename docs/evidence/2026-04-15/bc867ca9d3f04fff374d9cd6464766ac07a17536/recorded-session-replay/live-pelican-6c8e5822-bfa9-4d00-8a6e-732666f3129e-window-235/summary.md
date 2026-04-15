# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd603b557f17857772a54a908d5f0ba5df5b9405e501fb3b65a61cd496b30680`
- fixture hash: `sha256-a72ba01cb2a77634727f52aed3858de560e72d44f77e52442f91249de387c84b`
- score hash: `sha256-3b3eb1d07c1216e9c2dedee35106bdf65f86a353806cb789420e130ce3946ff2`
- bundle hash: `sha256-54f8c998e18880ad9d9197cddff995bc3fde2ebb9a5ab11e3d52de3450d7013f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf4126f9f6708337f7f0c62a2db19e988397589b870e571ff16dc3ae73782dd0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ed3649c67b2db0543b818c389a781623bec00128c649ec89f56585903710d8e1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17582bef1887684e3451040926a6065872b6e1ecdc028c95a86821849229bbb9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7d4b34bab07413d9fa69895fa38e77c5dfebaafceb22a0e2d9a0ac9e339adcf4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3d41158 | sha256-5d304b116facd28266d085cb32170dc2cec912fa1ea61fa072eeabaf6bd09726 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3d41158 | sha256-09952f4d9e7a5214221de0bba390f43f5d63d6a17bb8ecce784500b4a8df394c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c3d41158 | sha256-5d304b116facd28266d085cb32170dc2cec912fa1ea61fa072eeabaf6bd09726 |
