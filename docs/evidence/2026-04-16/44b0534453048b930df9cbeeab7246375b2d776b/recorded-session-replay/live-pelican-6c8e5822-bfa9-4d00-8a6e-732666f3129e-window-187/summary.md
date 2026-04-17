# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eae72c8906ce053ade6bf66b6f03ddc87f48a19f8e1b50fd6f47ba9774ecb440`
- fixture hash: `sha256-80de414b90b70f70f1d2f2daf70e3430dc27d1af7b593fd0e1e1dfcb61676ead`
- score hash: `sha256-bba962f2d02819b78ec7d5813231395ec5b3e2837d09b40ff3a6ddd13fd51f03`
- bundle hash: `sha256-98edbed7bf7e820bc439b30da311205a9802ddee345dc0ba2f51d77eb785cc07`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e2afd5c5e27e893dea21e62c6e8b163bef7241aac8748bba68a4d993b31b8a4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-731efc4837a9d44f7a637f2733efceca0d31607919c1fb829f140eb7e2129dbf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-62c7689a64c7e4032955ef7a2c7037d035e8e3f25600e61142e846f78d32eef1 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-463d3d34e7744364bbef4f8dc7e41a828e963c0a45e206cf252b7a94843a4465 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce6343a1 | sha256-2599e41757a346e54cec9efd7662975a6b5a0bc2918ce2c81e56494aa0e87ac3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce6343a1 | sha256-a89765c77116b0b3f6cbcbc28cba0d86f49655c80c8fb16c78274303e42d71e7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-77a18842 | sha256-7080bbe99343b8d9357309b6c14534708ab464d62a22eda63306af4fda073564 |
