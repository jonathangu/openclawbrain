# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3ae37b035b14582a9db80eca92cf9ea284e1a083da7e40c15c27766593c501ab`
- fixture hash: `sha256-c1a05a74d2fece7febaade02118c0528463204c7c70c4fc0e050990958f60a91`
- score hash: `sha256-3742d87074282ac175bd542b74cd3e021ae61e5d7b440b802f6ee72633207a49`
- bundle hash: `sha256-00100089aa9a36416167d02ac12a8f61b82d0a362a7878c8818d39f68a7e7e9d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5bfccd81fb07da2148e74d03332b298ae8343e32f9c89c9de3c815764af2fb42 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-22db9f14498b9688b1c1d773b50804b41984214f6571a4ed6579f32a1b3f554f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29578b91b40de252e60e8e309de652a8efce24a8df9db1340bbce3a3afab2502 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e13fed4ac721752f855304c5b84a405a9559878ac815c4ae95685ddfcbb75588 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0305e27a | sha256-30e4addbe502312d71a246c0d02a48ef713b9a1627a19c2e3f8d73e494304889 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0305e27a | sha256-989c2ada7d57a3aec9c36186d24148af83c753ad64f8ba5da8de6558b5d2ff9a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-bb611267 | sha256-c80dd9f1dfa6135111872661215dc9d9b6fbc0a9d4dfb256e0d44ccbcc0697f5 |
