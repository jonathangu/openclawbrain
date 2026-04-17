# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d8011faf3d69f32bcf0e92bcef735c94f96aebd8322b667cbe52a25917f1a6e`
- fixture hash: `sha256-f28ec0241ac4efd4c1f97733d381efba161e2d4c7cd778ddce2f415ed4529529`
- score hash: `sha256-d373da54e373eb963094fd8a9e53b50547b37f7b2b9fb8bb2332cc3fec6d00c2`
- bundle hash: `sha256-a30bdf6aba8d842b599710eaf3aa5b368cd414d0b2279389f0444b218887dbbf`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64f932981a8a1428e017d3b3bf8eed9c04a8f1b43e3be668df16d36de77d3b6f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eee7749a949d68f5af74497a354acc01624c430ba99767d4dc16abd63a9fcbd2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86bb48a7b9e311789a76f77dcdc139e46d06cc67c492b735f0839f999f1ca8f0 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2eb70e4b7103ac681f2a59e160bd8cee1fa8d7583d85a79f62836f0cf9c2cc19 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5ab4608e | sha256-f9820ac001b747201a21f9c80f3694c3e2129cf50503490e346089b43f251ed8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5ab4608e | sha256-eca389b80e13ba4b33041c2beeea04e427b2976ed890f1fb0fb0034791bdbbcb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c2ea7e99 | sha256-913e1a08ee039b0d8989e800035b96b3bf7097ca809cbaa595bfb87eb10ab54a |
