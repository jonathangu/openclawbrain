# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5ab4245c094a4283a2fe623f159ada638f3a2335ec988d52906db167e4a412cf`
- fixture hash: `sha256-200539ec6ee07f9053b46fde1430980f62e83407874931f79115b5f9bd8b8337`
- score hash: `sha256-96941ba34df4d4caad8303951a19f131949639c3d62ccbeae8e0e67f79fc7717`
- bundle hash: `sha256-d585939cd346bc6d74ab319f1faa58da543e26d8455bf8cd80dc1c5076690a2e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd1ac429de4ad281ade866e710d7bcaf6542300ac52809bbbdfe005490548973 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-858014ddc30df199c0e0a761a5dd344a30fb19f06aa9501890d5597a99913861 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c127b192452993b7dc92e6059d130aecb1c9d5eadd007f05b850bbe39395ae15 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8a6a6050fa0100eb294bbfac9572c41ce710841113f60ff98f81923d0304f83f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a129984 | sha256-9d072daea232df113fb8ea79561576bd28b9d747f5e87dee62f88ecca548bdb4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a129984 | sha256-0c71fde1894fee9eeb21be41eee92d640046e70e707c458bdf2e54160a643cae |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7a129984 | sha256-9d072daea232df113fb8ea79561576bd28b9d747f5e87dee62f88ecca548bdb4 |
