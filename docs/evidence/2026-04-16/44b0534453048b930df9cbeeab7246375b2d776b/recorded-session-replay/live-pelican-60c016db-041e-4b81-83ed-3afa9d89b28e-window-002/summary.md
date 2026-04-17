# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cfee0e7c7d7a332b7f5a4f24cbf19b04a7b31698d12d254a92d62753684d371`
- fixture hash: `sha256-539cc58588045f4d44638a17795295875c8ed45ffa9d4d266b2c19df9a95dd7f`
- score hash: `sha256-6ae31eb190e2c73fd81717c921bb1ba72eeb7e8a03324c6ff12514196a627c48`
- bundle hash: `sha256-ec4ff4bfe194f3ac67a6033521d7fdecb72fb2150fee3f87d4521248526f30ef`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32ce668e6813134ccd828363a96ac1b89f56519737480b00962b4a14175506 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0747c251ef2e0627c3b4240ad3870ad83958e282008e709393e64f9f82672300 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-01966710766fe746320233d508960c2aea40ddfcd64cfd500d6a24346cfc536c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ff2dc28070775f8cbd3eaa0ce975fe81abcdf5c458a4ed75ef3aa0b30530be43 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2a65794 | sha256-1649c880fd656567d37b1caa4b81d2785b15130a0328ff4d2af77e528a9b88db |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2a65794 | sha256-e52e88737dc64b370a60ca06fc19fd686cacef14386825571b6245ff453040da |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6bf1036f | sha256-b6ae154442ced8e52d1c37b32b1ddc27d140430dd7513f62d22e719b303cc361 |
