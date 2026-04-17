# Recorded Session Replay Proof Bundle

- trace id: `live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-44fbcb6576f6f004911009e236161c3fb072626b9bf71fadcefa3c9dfc1347dc`
- fixture hash: `sha256-178059882cfa4f40ce27919272b11654587c109af203796638882d20de0899c6`
- score hash: `sha256-30392c983f4fa9e0a395a0775dfc5dcb506be991d4f4751b43c2d0ef9cf87524`
- bundle hash: `sha256-8db19db504e2db9be874ee77da87c6a20693aa765da47d8d9b2a05519398dbaf`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc575aa7f7247c73c93f72af53d1bbeba87c049e551196e9ed2534df2a742d2 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-349994b2b220203a828f5c6541249d609d2f394a2c52b1cca834d6f2997261e5 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d87a7b71fcea298d4385b1cc02b39cbaa33508e267b9974276dfd171a66456ab |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-644b7970efab667ec0c7fad9ede0ab992776d1b326f50e409429d42d7f3aae7a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3f54a9a6 | sha256-48538b14f9476f5bfc1fee8a9dc8869967edd79f3e3310ddfebb365ae8e674b4 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3f54a9a6 | sha256-12e6a730264f96d84bc9c01152dbd2c4a1eba145c86b617f826ca33f41d6a622 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-34fe604d | sha256-c0b6ef13bc380136f4947a5737cd399069fe9a0b05d40240cb0551900177260d |
