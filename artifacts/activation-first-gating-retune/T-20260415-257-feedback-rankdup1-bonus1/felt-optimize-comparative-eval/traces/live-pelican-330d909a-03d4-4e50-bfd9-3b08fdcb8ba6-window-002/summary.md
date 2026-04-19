# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3188dea0835fbe3a5c0a4bde0dceb823e483b3aad858e66d490cbabd38ee5d72`
- fixture hash: `sha256-c551d2cb8b10201e8079d270837622ef96e1675624dead00decec0e3fb02a4b9`
- score hash: `sha256-7019b80c3393cc0fd40ec20d7046d61f8a5444f2e3802ed275fa9bff8b1ae969`
- bundle hash: `sha256-74df91dfbf47f5bb720dbf474fb9f1d67df3951edd489e7a24a56df9c7a88738`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c7e263c626ee395c7a005cdac6d8c14b4d8e92d0d3065cdc0b98a11e431231d |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-6fc40830a260d837d6838b5b5b3c383fc7d96dfba28da282b9162db5a7099a8e |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-ca5910022f53801c7d1c0e23747baef28f0ebcd6c40f45c4e752e554ee942697 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d1882b95f36f3e0b1db057d10e4b437ea55937888409cc5cc3a8fc4a0b98c506 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-9d5cae35 | sha256-4d5e4bf30ca13edd258354316d01aa8c7c88d0f87f92f885ee39eaed1928ece3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-9d5cae35 | sha256-9073bd01caf5796203aaf089563ef2d93d5aeb53f61fce73cfc9fcce9eaa9944 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9d5cae35 | sha256-2521b02c5daed468d5f4f71c0f068bf7cb5a3e5231c46185a33a92007d90a228 |
