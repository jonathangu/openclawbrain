# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca1018694169cc3fe531485fc537c09a6239e84c0e4410a019dba97e2a66fe7e`
- fixture hash: `sha256-9d6b96efb0f7a7d48de55af286c816bef6a9a27fdc8a979e0eeba28c500d12da`
- score hash: `sha256-d9575c2a71af508513b546dfdac04f78c15e6daad84d145437ebade959adbfce`
- bundle hash: `sha256-e1eb28a60f0c431eb4f64ca7ef0d4b11529c12e872dc21e98a01878db8007d1f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2225486ce356841ccd69a322b5b86cae51f3de0b57802b050f099b2bdb0a0f2e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-de52ec553fc156c495093148961de0d0d90c49be70deb72c917c94bc924b369b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2da52eae653619da6ea9fc7dd79741a328a10f1f15e223e24c72b1ffb97efa8 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-40de9528d14be2cff57e7a4889ee2dffedc69ba22e3bbdd841a1e06c11820d68 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc4107b4 | sha256-36e67009f68c2023131f93a534dcb738b70348781c0d69d9ff51f0811aad6710 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc4107b4 | sha256-3d0517f9473abef0321c7840ab34aec4c0d72c643f92917b4b41ee5c362f58f6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cbff5023 | sha256-798f97806af5fda7c7364105d370cfc48415ae355148e76fac929caa60d2b1a8 |
