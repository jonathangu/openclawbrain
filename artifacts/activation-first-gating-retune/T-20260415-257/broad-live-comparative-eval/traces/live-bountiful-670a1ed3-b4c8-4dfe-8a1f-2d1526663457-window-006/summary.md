# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-500bb42a51fe35739e28b1f6be3d9fe7ff92c6a8eeb2f053f3018ae2eba88584`
- fixture hash: `sha256-f69dca5c27c722f582ac3debb2e25adae4c35c5bd6a4749aa476e37eee07c7bc`
- score hash: `sha256-74eb85d195b379161c92f02c42c555482c64a555bb52f7060b1f79d37a7420e2`
- bundle hash: `sha256-627b10ff490a41a4b31cc37a58bc36f47db88b53387ad84dc80359ea4a6d9381`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a93c94aa4cb26ac67e3ba4bdee5fc22bb0276c3da7ff11089c43e42405c272c |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-10c2c320a4ec2b8efc982b2ae36781267d2a66f88215b964184f292e38d77f2a |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e7a4ddc0a425921c11785376354f3f67cbeff10cb0d6f7be7bb569702440008c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2884e0d1cc89d894501861cbbfa59fafd5d11e0ef46566cab68a20c6ee0aed5b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-af73c9cf | sha256-9974c61bb60d9ed321c56afc399137a64b3f7ded600859c269895b2f2b8bfe21 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-af73c9cf | sha256-666f8ffe7af42e4b0d19921e1b48a838177694ca72f0d9ad97c6540525ccf5d8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f195a9c8 | sha256-1136159de1071911f0cd0173ee23ac579cc47f2ddadbe08cfd783838e9863e6f |
