# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8b83288abc1a5c66a218574e9a089abcfea75ee1de4f5813fd07c339a4e34fa2`
- fixture hash: `sha256-d84bdb541f6a2d5c8236abca3a843aa21a0e1c20f003d0fc5eb1d79b307b698e`
- score hash: `sha256-cf0890bf8204b375527d7ad0cde53de802e59a702aef805ac71bd63dcdf2708d`
- bundle hash: `sha256-ae61e7aa104902f8e17aefa22da9db6586ca34fb5735d783897c2c0417d6f36a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ad0dcf523c4d76bf7e5aa9a9c949e660e04aa89d0cc57603f9d8d3b2165caa4 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db98ca2cb667045373e1cb2ba5404dda1ab2b3682bad391403aa923115b51df1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3616e88996bef753609ac5dd5f025c4addebd6e47b18be0c3777ceade3282655 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a1475427eaf3d8844005715b777e37e2ab50a25ed9ed6f19ed797ef5b70028d5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4ea75339 | sha256-4d690f203030ebc008da7b1657b4d93351e05228b41c52581b1e0bc610e2ff0d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4ea75339 | sha256-e56c74f173d6da19fe9179324cd07d3f917554376ce8813a8b83e25014f23bba |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f3ff82a8 | sha256-f2255c01f61ab4576fa6d91a7efafe72b7d3e624c622f609e2cad0636a6f39a1 |
