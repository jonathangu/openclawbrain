# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8fecd38f3aa3470c67016a58c02da538613366240f311d73e765e2e999bfc5e1`
- fixture hash: `sha256-9a635fc4466dcd1f01d2e94228a353c7c6a97d36b77eaea2bf2676d0c4e0cb26`
- score hash: `sha256-9f59a2be6a4cfe13e973edd583c745df6431791fa1908bd839b1118de47d6eff`
- bundle hash: `sha256-7479f7fb5fe3e0a3174bfc0cb85b87d8060af39290f007038489631a8a34ffca`

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
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32d500730de7d73f2a2bf38e8b78d2d6ad04a3a58dd8029622c951f7ddee70 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2450bae449eebec6a9d9ca420d347d278c9c8704095254702c8c377a268218b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-affc39d614c22eb88b196d605eec2d66031b365a733787e9e967451daf81dfb5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8910e17f730e491c6c99f8c3845514debfd640340a87f22794262c4c04612bfb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6afbb0b3 | sha256-a02f82a300045d49a4dca86f832fbf61f4f7263c9c1d59b4a92bee3d254ab4a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6afbb0b3 | sha256-fdd4ae524805a8b4938519995144fe79ff6564aafff404e17c4482c14b8d0fb1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6afbb0b3 | sha256-a02f82a300045d49a4dca86f832fbf61f4f7263c9c1d59b4a92bee3d254ab4a8 |
