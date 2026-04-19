# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a8a2e6a63cc5912fb58030e76267c771c6d07671775935e13384022cf8e7c59`
- fixture hash: `sha256-d3b9199b3d1fba06ec6d727611496f93d92d13e1e28ef25defc3314d0f80c421`
- score hash: `sha256-88e61758c36314aec2d9fab4ba3945ef8110e43bd8466f174fa867b73dd7645d`
- bundle hash: `sha256-17260870c9e57fb82bedc6edef90ce585538b99f568d34e9d1f8e692b422abe4`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39cca038bdbd32b11125d0c6fba3b1b3a673e66a982ba05e8a320b541d748401 |
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-095d6109d25241632dd04ffd638c65d2d6379617a823d432f5acb8b7cdda6019 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-7b2e051c3bb0ac3097cb84b8dbd303675f5d24ea8df220e4bdfb2b998341eb0d |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-2f08b9b0d0807262dbf7ed6504ddb67c21df79c1e447376550fea4dd9c6b991b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-06631aea | sha256-a47dded75d8ef27ee06dcdf6d179080f74128c701301bcf283a9858408d81500 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-06631aea | sha256-46c55e9556c87cca2880cbc1527083d3bc17c748ae161fbd16e5c6c89a887a5e |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-06631aea | sha256-a47dded75d8ef27ee06dcdf6d179080f74128c701301bcf283a9858408d81500 |
