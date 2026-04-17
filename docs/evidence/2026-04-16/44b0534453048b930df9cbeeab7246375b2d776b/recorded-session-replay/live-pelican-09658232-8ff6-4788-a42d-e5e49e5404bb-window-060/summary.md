# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92040fed208ea65585f475c24b64fc03720a2a86a8c84eeb65240f8ebda78b47`
- fixture hash: `sha256-e789097b492b370a3cb207f40a7a3a195c61c7549ef2c7d39a0e569e0dd15633`
- score hash: `sha256-54eb048e146142f37580ed1c46a39f24d87c7a2c04d1a7fe4e5663fe5864b69c`
- bundle hash: `sha256-7b90307b855265a8237ef0f2a1d2c9c7d5071cba757ae2c7a39249695b747e7f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-648eaa1db4d7048b0d51fcc33cb635f35068c6efd52393b35ef8355c224bb749 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a631159c3d8c68800a7898cb69f2f352146eff5c6fd91c519369cfddad1ee2c4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2bbab59b5ee6f91f5b0dc727e73af45aa297d96a0d3c48522db7638893779ff2 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4caf625de13e0519881329d6a4c453aea1da3de2f1d48ca540000c62623a7096 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-57932616 | sha256-6ee200bde834456b7d9c2cf015972d4858d793def85ef1d4de5528bbc4381dcb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-57932616 | sha256-2dfb05ff4b4e0e4731501e4463a8b38ada6eef3d4e0db1b699323d39291b9045 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-71cecd1b | sha256-f60b74199b96097bac568d594034a2af0a498c5292cc0b220502bdbeb034bcc7 |
