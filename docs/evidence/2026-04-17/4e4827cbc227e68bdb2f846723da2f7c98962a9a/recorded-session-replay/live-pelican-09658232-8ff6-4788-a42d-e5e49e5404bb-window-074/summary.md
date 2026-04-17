# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-145181a4c88521ffe27e5e625ec9bfb16be2cb3d2184f357c331627e92abf897`
- fixture hash: `sha256-bcc383a8935099d2c1130fc4c95751549995a1b863da5b373b03632cf18f4269`
- score hash: `sha256-33bfee2a0832541913f17f5d14eaedb00390dd033376da390cf69cabb848371d`
- bundle hash: `sha256-85e383617d456d8479f7507b8eede5c0c4453c203babc4b74f10aa35c5e86145`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4afc95d7380c998fcb03d6870fdcb5302e0b5153c1f87770cfff10ca04ee8cf9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-843eb3e8ea8936c4e487b4fb1e0fbb15c0441b537292002fb8ec0e8e7e13025c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-89e378b7bf9e8f55a7e6827d1172a6979230a0db70ad9b8cf8b04ea844daabd0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-10b0107a8ed58cf15e76d3ce6d7460c7b869a0bb9ea7251dc5d82bbc60c2c73b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-197b5740 | sha256-83ee1cace339a16f55663ba10e10bf9602365553bcd09c04f81d8e567027484b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-197b5740 | sha256-5280ae917f4c1d2de06fe6913ee190f5fb191543ac2479fa96f4d4eb109583d1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-05de8a61 | sha256-32eca8425d9eab236652ad52cfe155ee0df3406c12f4dee8ca0dae55e46275a5 |
