# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-945c2e59668de577944ec7fc5dd5f9442c630538679d6020f6fdac64e2a21a17`
- fixture hash: `sha256-19b533ee2cadb7bef94e2f868a3d98284f247e98f26920f7fea15136681e3d11`
- score hash: `sha256-ed1b30a87cd09f268c4952b19c5a84f3c084cdaa7053eae895e1b66b9ebe45ef`
- bundle hash: `sha256-c0d8509e5b3677c33a7a4f6c7c724856a9b5df36168ec3b285fd47d7591706ff`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d37994d9aa92d2e1c7fd5cea54b3093f268f662f580c5608c088fa86597acbc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-472628e3e395c9b59f1636d274e875ad6a1e05e9340d38a5256c6f02dd812ba0 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-584338ccdf73c31e00816e79c3aaa6fd2550504689928f6382e188d544a1ad21 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5d3a10b641666d10cf7bcec87054905f5fd478f592b1b804b71a125a193f3576 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e46b98e9 | sha256-984e94399678104c14d317246db4d006406c97193b8dd4d8656f5053acfeadc0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e46b98e9 | sha256-62999734fd6ca658dffff29fd742978e1537875336d80de6a40bb6f500cf05f4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ab6b8276 | sha256-8a0949804f66d0154af3700779551c8d9ffe16dc10f64b17559750d3ac6a4ed7 |
