# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176`
- winner mode: `graph_prior_only`
- trace hash: `sha256-480f373d763bedd2f5766cb9a1a8860701112223bd910911c4830c2fc4277912`
- fixture hash: `sha256-2055d04a6856d7cf43d112e858be3651b8402ee14faf73331e6f59144245384e`
- score hash: `sha256-13f8b40885cd6c4b2d0601e75e3ea4ad5d10b83751993cfcbaa7dbcf4a00b8f9`
- bundle hash: `sha256-0f99b9af31299021b4d329d10c3edbfe2eec4262e22fd824836aeb2c276cc2df`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cc6a8a76ddb7a25937feec38e19ee175087db4867c24970d2759b39f1c9b4bd1 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-01a07fae77495fdb1bf6bb96e3ea1a5efb060a70fa25d529b619897f35be18ca |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-67b9b475d35e55d37327eade5a4d1640600741e12bded18f190567c1e2527f8b |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-062f1254f4a0667fd1e69088db2e0598676919521f286e86a05603e8c3de87e5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-076a2132 | sha256-94c3152b79fb0360e550d206aa0891b935969fdbb6bad08126064fde9548f80f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-076a2132 | sha256-5dc85394d596e43a4e9461bdf7b04a04fec0907e88899aeac9f22894ec588dfd |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ebff6ffd | sha256-be84cb45c501de5a22ad98a5aa1dbe2b9da8e2f27bf7a01498fdbb60913750dd |
