# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15894236758cffd6885df088771bc9158a039d8e6dca7ba37e0c0ae93f2bb22c`
- fixture hash: `sha256-897b7fdc496e16305fc54601a8aba44f23b5322a6b7036c26e9f447dc3d9e950`
- score hash: `sha256-c6a286e01a841b0087a98a6e860c62574d615eb9c74dfbec00ea03ad826ce637`
- bundle hash: `sha256-be7ba3d9c10100bafb07a9e6de80066cb55a1ee2ad3e625f5fb550866a054c1a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05b9f1a0d0ad4a80c5a15a8f7ef9c5d2527f8753fe005026d39ad6af8199556b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-19c9394b194aa0838ea70ce134a2c30dc4ee82ffd965641022d705c0a3161e04 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc78ae9d3d0ce4045ade3137442b04863057b1f833e2ef2094ddcacdaad44031 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f83947f8c4c0474954654f7133d2dee9a4e29c911d8226e632968ea2a3610ae7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cd793b27 | sha256-9a1688ee01bd442db2b0fae78155c80a782a4fd9c5cc5fd960d684e89f80bf04 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cd793b27 | sha256-40107bb35b313679e183c736953ce83eb47c4fbc11dfc27be67b1b2386d0d632 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-91f9c5bc | sha256-b9615ecf77395c0f2969c178cb363f4ea8e309b0de2bc37ee76ef0f049bf54a7 |
