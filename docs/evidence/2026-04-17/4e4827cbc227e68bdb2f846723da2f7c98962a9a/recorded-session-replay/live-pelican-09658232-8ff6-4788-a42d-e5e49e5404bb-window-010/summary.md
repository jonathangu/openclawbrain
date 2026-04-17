# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd69cb2bd54df203880c8fab0fa4c855341f06ecfb9d6ec642144558419aa71a`
- fixture hash: `sha256-f29630fbd2f41b8d395fae06865eb7778e00433b1298788381332e0703a42702`
- score hash: `sha256-a183f46716e275c60c120a93103e07b06f985110f9a98a0ae260f58f3d82a8d4`
- bundle hash: `sha256-8cbeb506f95c2f91f09e58ba771d350c439a0da3a10a1222bf591975d6e5d21d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80e65d944c503b7cf482a4ac157c70bd9810fcdc3cd3dc77c36042f87f3356ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3e1187266991a9750d076f3a6fc3ddd6e448c56caa2b9207a09f5e0dae0f420d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b7606c1005bc9586ba4062abc5f3bdffed81ecbdf68735f2d06e14083d987814 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d86757c051024b63ad773f36e524f91af8f38d91991df95d9a0c49c7bd74c455 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-06299334 | sha256-5c5449bf3dd6de5f5358113750dfaed267725108e369b32eada90e20a5162912 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-06299334 | sha256-7193329cebb44a2cc1c169aa8aee44fc5527caf4b90ddbb45f4d6a10c1b0178e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-413884f7 | sha256-cc27a6ef95868d3cbd34b379152e939565ca7cd7d89b95286b5a1a3ea4a490c3 |
