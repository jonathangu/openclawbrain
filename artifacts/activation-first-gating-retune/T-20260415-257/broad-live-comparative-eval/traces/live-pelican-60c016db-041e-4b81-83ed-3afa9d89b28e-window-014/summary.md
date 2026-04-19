# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adbfb582784ce9c57067bcd682b42040f9ff5a4fc2a41a6b215fa1e5e63926e2`
- fixture hash: `sha256-1b81b9ebc5b6e57a68ac36d63b63963fa7e0e03c9b05269658a97fc89e8025b0`
- score hash: `sha256-e894b4359fa68410b14ea80e0464ff8d3564a617bb57e5098b0fb40e0f67adb3`
- bundle hash: `sha256-66a9a35eda15aac808907f6eca4646739c18d176e06e9965f7ad157ace664c4e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d918c89a43fc84e7a627af305c1d796a487842c9f1cf040b6474472ff6068ba |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-5883580e85141faa0988f028202a9aa11d1f15a61e5631d9c270fdf098fbe5a3 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-007ee605a01ad0f910c7aeffc621800a1f3aa968ee8155b1b2461697a304ed40 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f61331c7384c1001f4b6eab1bdf4581485753b3c96c209ec19c1e1f3ec41e5a7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-71f32012 | sha256-7c5d02f842db0661a92f4648c0492ea0b7deabf19801f1aca8b33edb3f165fd7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-71f32012 | sha256-7dca7f3e0dc6320f28fda937c6525f8c28e10250489f3a138715ec38e53ef504 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-71f32012 | sha256-7c5d02f842db0661a92f4648c0492ea0b7deabf19801f1aca8b33edb3f165fd7 |
