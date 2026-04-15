# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eb4ce4c16a0b4086f9bf16153627317bccc66c138e3f3eabb740de5aad356d3c`
- fixture hash: `sha256-f8df6b8b0d3896e4d68df7e66273fb59a221dba4842848c8bd3431e1201171eb`
- score hash: `sha256-99d6c282250eea68310ee81755bcdee6847135b123b01343e2d3086890888db2`
- bundle hash: `sha256-d0dbd10f844f58782d4f6b3e3c6d4055b9364e069ad1c2d361f9820c0492481e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31a831566ed58685dea8a0c35a91e51999c06d52779d7820057deddb5dbf99cd |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-63f753f9a39e2988659152624be979a3e4195aa8050d5f74070eaa3119579356 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4e9fc4542a72207efe6a4eaa912a33a799e91a2937217edd482937bb090b189b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-00f3860849435eac99ebd1a5cb730a139f70317efeb2057f8786ba50ac9a614e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3dda75af | sha256-d3569052602782a0c601121b89874f1c1bffb438a78e690495036de3903df493 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3dda75af | sha256-110ad60c54e9d838c2a410606d90e42ba877db149be4a77a04243fc820394c1c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3dda75af | sha256-d3569052602782a0c601121b89874f1c1bffb438a78e690495036de3903df493 |
