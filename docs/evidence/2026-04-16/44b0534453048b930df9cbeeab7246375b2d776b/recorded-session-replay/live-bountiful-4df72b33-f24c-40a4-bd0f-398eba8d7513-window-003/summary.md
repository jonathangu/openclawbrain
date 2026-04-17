# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8836c19286d7dfc7e25365f74f5e0786007d5f48b08d8fcfba5fe79b0f03c2c`
- fixture hash: `sha256-0ffaff36365448396a5594a68d8364ec6eacdae9fdbcb2693a4ddbea65547f4c`
- score hash: `sha256-93a5defdba8f00241dff694061c89b52607608403d925b03f4f93f878685996b`
- bundle hash: `sha256-270b3a0ae225f2e7b7cb06ffa8d3c6f8996a70ee026cffbebcda5dc411f9b54a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4e67449219060e0eaa53a64e9ca0f2f7168ec707e126564ccb072cf633b7d0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2790de173230ab8896791cbc20745f0303b1bdd45f35052567f8251feba72b42 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5fdedf0869fe0f4a335a9f231976338a18cdfa9d5c28360473ffc1b9ad202e75 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fa87ba21f97705d1c287ed59bd6664d7393bf308d375ecaebb02a96c3b926a8e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-466c2241 | sha256-c78d2798a61c247ba64016ab15a1accad0a530821dc2ddc70743cad55fb9eec1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-466c2241 | sha256-144dead4f40bd9396268b03ec497f8bd9e5a77d386cdd82e57fa50dae3ed36e1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e98ff616 | sha256-9c9f077ba444c459164119050350535adbde4e1fd0d948d1ad878c0dac9b11c5 |
