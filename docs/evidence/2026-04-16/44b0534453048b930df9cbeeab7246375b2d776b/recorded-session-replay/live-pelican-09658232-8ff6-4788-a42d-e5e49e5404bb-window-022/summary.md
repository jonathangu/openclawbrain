# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b973c64c8eac5a0b6deba25fbae9f31be4599e3d19192c7c9dd0b18e718f1e`
- fixture hash: `sha256-b932d5e627b7081f980ab111b252e205aa7e0185bfcd774e6388fb9e948098c1`
- score hash: `sha256-1be9e9f185bebe61f274a08f7e612a12ec0c52bb7bf042e553402154faa845cb`
- bundle hash: `sha256-8c08352d88ddf27e4b3f47cb344d826e0655aed778bfcc3dc518356f6f81be87`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd3c28bfccf2817f3d01d14dc16c97875abfde806e8cfbeff2d04b6e2a397e7b |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-255bb896e92165569d1363d8a8caa646237a1d95333d1b6de56f214b2fb1c4f3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d61f569acd5a3351d322d9e06ffdd3e418c906acc4ef5e154caa2fc32cb26c1c |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6bebb590f7be5b50b9dd792467bfac91df104be4ad7d9ac6d0491288ca48793b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-430ab660 | sha256-0a467e7586a59bd7d7d15e769e7abdd5bf0223bb8cbb72e70c4a3a5fead47005 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-430ab660 | sha256-0a467e7586a59bd7d7d15e769e7abdd5bf0223bb8cbb72e70c4a3a5fead47005 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-18ff9a53 | sha256-766c8f52877366a1a40329a4abdcf16f7fbac8d9884a31855882d5c00cf4c17c |
