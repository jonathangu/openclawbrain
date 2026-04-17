# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2d785c91c6c2597c88bfdefe91898000c30733ecf3cca8e1fa5fd2d6621049e4`
- fixture hash: `sha256-63b7942b83cea800c5fc9cb957ce0307322538d9d8e1a745ea7ab80b74e65911`
- score hash: `sha256-2164028857c9df71eabdef2b1bd99aa30ce7c942c4f692db5c76ec28b3a49569`
- bundle hash: `sha256-dbf849fe71b8e09f02bf306c2f4ea26cc4a62b98275a0c510cb86b0ddb6d8538`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4648c104e20ab98d8928f41590949536cf65a6240f7fac95811ce6126bd169f5 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-62cf5a74058298823475fba5eb3c5172967503bf0facb3b81026374d79e7f40b |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-b325ba9b6346fdcb4dec5d4d23669817dcd5848f9670a208d4bdbab90d99172d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-62a8cb64e055396a556ceac3b55b2e73f373d120d54a222d50220a51eaa10ac4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-f88ed3be | sha256-f398a8b9c50ceadd6be4d44f4277a7b3d6205a053fc257e5c40d0af014622266 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-f88ed3be | sha256-b8bf3b4c382c12d11e12eb600f8b99365c1dbb6f03eb8ff0ddb96fd05722e8f8 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5ae12fd9 | sha256-f2d12b7746d1b8875e912e2e65801900219b566b72964480428c36ad39277f57 |
