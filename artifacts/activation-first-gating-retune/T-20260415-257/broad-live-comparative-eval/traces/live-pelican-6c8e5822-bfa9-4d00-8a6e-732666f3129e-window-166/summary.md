# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec7f0ca39d2ce8e4aa075852c984c14df45efbf7ebb099adc4d8318c646741f9`
- fixture hash: `sha256-1eeb0e4e14f003831776523471001891e5f51483edf8cd0fe82b3b2a7a4e72c2`
- score hash: `sha256-86b48e975d530fd83439f87a0d57fa01ec0d34e6f0fd86d03d898623e4e9a89d`
- bundle hash: `sha256-a9c2d65a4a1bdb334d229a1daf7bea91d742953702d66669e4732b254e003c9d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d39232e9e4182be91b475d1dc774e142ceab1f9213fd98395428e4f29aee341f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fc1bdf304cb1b81ca22a2df95aedf49faf9f77ae834df5e90ed003d57646ea7b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0a6add36173de0e860ba8c124d303ea334a259fee36cfc406aaafee3de851d23 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-40f3bab5d63ff5f1e6c179133a93ba85f1a1d6256e39b8ba774fe94fb617ab86 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-deb95161 | sha256-e4d9204e2dda06c4b83e6204bca2ba96b4f0d8143f1fd2e10a42cc6f1cc8e095 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-deb95161 | sha256-944f88d5d9a066c93bda6f2ac9f33127582346b6daf6217f2ee18d717b3559bc |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-81b43aee | sha256-23be7513e4aedc5b929c6b45c3ce33475a188653ac5f7b3d4d2c65bfb93e4c47 |
