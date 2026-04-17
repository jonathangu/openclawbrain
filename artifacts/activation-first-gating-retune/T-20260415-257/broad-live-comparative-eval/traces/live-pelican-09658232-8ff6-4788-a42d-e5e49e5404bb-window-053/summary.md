# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ebe19ca9bc459ccc52f05cb3ef8e24277b70f060cd7939f534aee99c63488ae5`
- fixture hash: `sha256-83e33e2dc3d5736fb8b475959b3f799a1522431a9ceb8bc4c7fc74edb18967c0`
- score hash: `sha256-52fa3e1112681b5e9a8d5b1c04ebc75af869cf42da72cecfa4453d91b0eeac7f`
- bundle hash: `sha256-fd6e443db3b889da46bb7a37a2139fef5743eb30597f4617947c99f4b4068b05`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38dbfef9dbd8d3a664a1f95db7f92b5a52579781d0eb6bee8aa47758b54b5ce0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b206625bb84139fb5fd75ba5dcc4a8c9e2cef4cfecf7d0aeb0854cc066f7ad1b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2d42396fe7694460aeb437c149f80bfe46a357db5aa376a90127b5461e7e521 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b6a127d818a71a76668f45ab62d6008b98540db94895ba61d7772c48d7e4311c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a3242dc8 | sha256-500667158564b0917372323f55b9250539816c205c3d37aec03e56c38b82fba6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a3242dc8 | sha256-c94061332781d081d10a940000f1697fe86a007944278d96d5e3ed3b98978e80 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cd5fd28b | sha256-2a5e1e7f56a724579145825c4b696143a29aac342cd2a9ec206a4894c2c160a4 |
