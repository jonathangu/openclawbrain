# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37b9967646ced8e1a7e53e66d95e96c0d5cf9872e9f6cf5f223ff75c45212fe4`
- fixture hash: `sha256-e5562aca0bd9165edb9d4f0591f9dae6981c5299e9b8cff4453286d3a3e6c950`
- score hash: `sha256-1f7a1d7520b9ad7c9eea398c353c94c8e8b67ad9171ff06e80f09ee305d5d314`
- bundle hash: `sha256-acd846a0bdb0e56998aa3bfb147c03629c6f2dc15c8b78a45028eb909300ee08`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e59e0368806e0012160cf4b2dfced7c5e08071a2c01bb62268694e031a82feac |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18fe68186433a61d0434d66103e2ecd66b850debd130e923b8d45bf7c4462335 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af3af10a37ca8010f1facb205455eba402eeac6b8e13ca0969f2a59d0a774c81 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6e8c94a0a96feb60c8c88816944f70ee12411650cb0dd919e0d9f05a4136bd12 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6965bc48 | sha256-6919480a95bb6cb595890072af89852676aa143c7867db7c3e81b21e4b131f56 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6965bc48 | sha256-4a469b12dae93488631df4116b4ad330471a98096a09d17a8eaf37cb7faaad24 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9a430ffd | sha256-0329ba6243c91c8132fe37259c1a487bba0ef800cd8cf012609b12aa0333581c |
