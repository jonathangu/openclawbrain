# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b0963b5e18b40bd244437e7e0701feca11058e18371c7c7830706c53a28f15f0`
- fixture hash: `sha256-ad1fe96e9866fc7227d860e828f71679e46d996ff90526f19b5279748e32ad9b`
- score hash: `sha256-b84935cca1de0e087bcd7aa420427c13793ce0daccdb7ae1c935494e0832dd22`
- bundle hash: `sha256-ef9c2862b4770bea0070abcdf2ec5f08218a7a1b949ca38124727c88fac6469d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-63768ebf632f25773108234ec4f8850307fca437412854c0aa69b01f97c1eac8 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-0d68807e0259225de10b156c9295dfd30f0986d6247e7cf6771a60471d3bcc09 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-65c9aa5e0c33c19563173fc025ab585df2d638bc8c27f709dbb6450027b2645e |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-78b364eb0fe9fbc738f9912bd79e40e011bbfa3c9003a047e4b52c29cf50b907 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-91ddb890 | sha256-f1b4ef389d60d6a9fe3394902555a3d76167d8a341c5f0eecda34097c6600df6 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-91ddb890 | sha256-7b704b6de67634987c2ffcd3a4a90ef79d65bdd0ee2e4c38bafcc7e9a370d896 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-91ddb890 | sha256-f1b4ef389d60d6a9fe3394902555a3d76167d8a341c5f0eecda34097c6600df6 |
