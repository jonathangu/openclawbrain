# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ea104683f4faf6c43d1e2de1d4eaf420a688cff5a862d0f10bcf59142dc68752`
- fixture hash: `sha256-f88768ec722911fceeee6af7386980436f1947b9771d5b78128543260a9fd9c9`
- score hash: `sha256-44077c424629d62336c32427d753cb7ffa9ad1fe774536a27a1103123a4fdc5c`
- bundle hash: `sha256-93fff2f75f70104da4b51c837f3626dddcf192c0e3ce209e787e4fbfda6b322e`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-01f110226bcb696f2d92c204531dd80bffd0df5e206ead0910da9d2b251a70cb |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-4509bafbc413a94bfb35b50c8687a3dca9c8bdfd011d9d773f1f0cfe55ad870c |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-2040b0f9e53a4680ffdd13dc75b5c0569e57e32debcb1434c8993c0107cd63fc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-20892a884e60badd614da0154330d17b1a11f7200bbb4321d9743d412eb0f6b6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-9dba94d2 | sha256-0fa133004fdb8a427446912d26451d93d0936ee74dde7c47ce40fd9a26b2ef3e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-9dba94d2 | sha256-7e8d48d94b3c427e97be4904088cccc034899c0d1463d939544de24b70f57efc |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9dba94d2 | sha256-0fa133004fdb8a427446912d26451d93d0936ee74dde7c47ce40fd9a26b2ef3e |
