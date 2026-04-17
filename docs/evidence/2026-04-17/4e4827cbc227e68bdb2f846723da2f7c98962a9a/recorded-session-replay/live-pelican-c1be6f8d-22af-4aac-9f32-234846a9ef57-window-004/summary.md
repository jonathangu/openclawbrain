# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b9882f49bf30cb6d948087b310dd1f1c8c43cb51ebde7842866360d6db046b12`
- fixture hash: `sha256-371ddc3cfed0332b92f92e9c2b214fd34bd05f438837cc6562acfdd4c1e2c749`
- score hash: `sha256-2a4c63ff730471e95aee77d8730c901f6721b8dac3443722cf095068794f1c36`
- bundle hash: `sha256-f86533374adc2f20a925c346f6e6925fa037768150a5cca1c996c0e67d2fad69`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d3958fc805f3776c38e6e687c85563bf09e68cc8dca03392a973d72cef995c7a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b092cbeec1b68223a536f598a90ca123807df5752a57bbca808e10319885b871 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-301bad7e60113d323cb8283f00310fe27d901420ad124b29b88889a2e11d0a7a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6f6728b48b3601867d9b6a04d1cd62faf2bcd88c21f9b5b434753356294464df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-77b8cdc9 | sha256-7e6d0cdb0f5362d4dae9e6f166d3d6c3ed09c9d2c614dfd67d5055651448e89e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-77b8cdc9 | sha256-984d305543a0f0cc770c08cb668fd14965fad8cf17d1b3650aeaa9e97c76a53d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b7e13102 | sha256-26f5daa2b07292342a0cac0331604909909f88880189ada1ff5a0b37ffbf885c |
