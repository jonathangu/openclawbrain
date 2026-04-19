# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-02900a2958e928474b222f0f413d849223d64080e597f6ab14afc07d8b01433d`
- bundle hash: `sha256-5122bc93cd9f0c81a72cb46991e17dcb5b55fafc50b84f35c1a61a16e592064f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1623aa98a961961a182098cbb09dfbf96da5584b9efee0863f57cb38d7ebe41e |
| vector_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-247895ea42cf6d388023e715589601a099ca99ba79d9a6ec05d2ae2eb8371417 |
| graph_prior_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-8a412f23f0aed68347ab5663a7c0e4a42f8da967dc67441d9276412687910278 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-b9104baf584d5721c42948e7723f42ff202cda1e456871957ef2d8ac581becd4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-13225c98 | sha256-4e31b3931cdcd4b2c0eadfb97c1ea2c7bd56e3fc40340730da0adb48488edad9 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-13225c98 | sha256-3c09498226c896f846d83d225291932b102c846a3d0116dd3217a62138d52422 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-13225c98 | sha256-bf6045700d2ddf641cd5b66c4025c078a6c2bd5458c5c2c02a16988d274786e4 |
