# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d855fd526b0432f6da4ae83a914585ce6467161a22fe45c628b20919e2994b08`
- fixture hash: `sha256-a059e9b8611b556f3c483b97168ab252147668d3316414532e38d0791f5cd0c4`
- score hash: `sha256-7ce69506c30815b2a11d6038a6d2eecddaf347cbe07291fd02232fea6ad85053`
- bundle hash: `sha256-a608be7c5a437fd844d411455559d6d4f6b57e25c02bc154b037f776e0fa765c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46808c7f90eba103441fec044b9224d9dea48b85cde7d0c53efec734a800db3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e0f24c526609e4b9f7ff360be62d6bc92a099d677a6a95efc87d5447052fa2ec |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2efb24d7f91847d552940ff143244a06d484676283c78bdfd8fc4cfb25733ad9 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-dff3b5d57cf5e1a4f60b1a044e148d1dec4c8a5276236946170a5f3acb2346f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-239adf33 | sha256-e80793592098cf1f51d401e082a222eb0c2b7c545784ad26c512b574fcde9291 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-239adf33 | sha256-99fab48a7d7c58ac878831688f0c973d6bbae210092b25fcf31226b2b672adf0 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-817c2304 | sha256-cf131e7698b1f7ae80a9151553e1f75e97fb0c370aaf45f8622a239d921dd1b0 |
