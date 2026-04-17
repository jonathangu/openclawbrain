# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c4b362666455e64ccec1b7026696e6d7ee86b07af9d91203554a5f880643a7f`
- fixture hash: `sha256-ed322207ac696cb8afb94d5f75c53ba4423b96ed55d3a35abcef96ba37d6147e`
- score hash: `sha256-bd32eabf4bc57f98713f3b8e88c7da07051017f910dbbafe2817116b58827116`
- bundle hash: `sha256-0fcd37b3d7cf4f81508d605829f322716a741ffa1e6dd3d26ddd5ffc23c4562f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-659f0a773cf1e71603ffb20a5a28aebea6d5db6139dfdf0255a605e7868cd22c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-973030f315e0c1b1735dab026adada3047e13065bf3f0bf6bab4976f2e20f747 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bb66d8df552366bbdf87336a7ce829bbb354b47a5fdc688e2020c6eb0d077d1e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-93f6847e1097061d726b894d4a74791d8cc43ae82d8c928cf7f0620b5c750952 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-786ba792 | sha256-0e25fe1199d303ac078b4fee3de666eeeb97f40d021d1115a4bbc87a23a23f45 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-786ba792 | sha256-376ee0b9323e82166ff8e2734abf37e7fc5ee5a8d7270f322d0707b666a983b2 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-46b97049 | sha256-fbeab764009ad3b3c75782f3e46735573163657468d56009056755b05a56604f |
