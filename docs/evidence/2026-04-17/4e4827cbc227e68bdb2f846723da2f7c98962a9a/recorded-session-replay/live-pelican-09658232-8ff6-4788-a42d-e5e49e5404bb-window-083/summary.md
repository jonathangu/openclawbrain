# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84b9a4843680de911479c2420a8592984c3d84b3d54d06debdc96d5c918ea030`
- fixture hash: `sha256-be373dad3e692162d5000f12580f9371232c68a9b0f09d3136130b3fe2a640e9`
- score hash: `sha256-05e179ef4a6dc89ade95f14dd8e4106116ba34b356a3fcbe72493965c72b7354`
- bundle hash: `sha256-0c0c1f671e87e27e5d831b63782f73f42ac8d0d66eff2cf814ba5292ade550f2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-34864147a65f338d5fe87baff27e70ea8462feed84ac2fbd4644ab5e3e006364 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c761e3c39ae4777479076b9e4d81fc70ae0ce24712c6181de9d15a67b99bfe70 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-75de447878102dc11bd77e63308b34ce47cad4c423a4f1b96265f09b60efd395 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-edb779b97322d4bc68b7c9181e7361994b8d8c6662a5d2375aa10ed2c1f30125 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-88404622 | sha256-2f7ba9c2855be37ac1f0c854aade90eb47caa907f4afb556dba206cba2db1c40 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-88404622 | sha256-f35c9fa45540d3924254bbf18be7e2031b70d9b8f5b1e19b8b5e4c870b6d4200 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3b967465 | sha256-4b448d24bd210a3e0436b78f4e6343bdc36f0e7c7f84bd8d5026e7fc7457f580 |
