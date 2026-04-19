# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb2cad08b75a4c5a17135f849a796fb61b6aa111c1915e1e3fbc43ee29768a21`
- fixture hash: `sha256-84cbdf3fb211d244e4240d521b9374e489cde8de517963996815de9de3b7ff1d`
- score hash: `sha256-fa6659678232da567fd0a1194e8ee812f41b4c842b7d417e96bd9c7eb05316b9`
- bundle hash: `sha256-5163aa4ebe0357217c50bee6c07d4e766815eda49368cb42f732167236d2f5c5`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ed7632976503f51e965dfa45a2710170b9804510f9f71a0462c48d22b4bb5cb9 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-210fd128a377d5789d40773680922b397ae9e4a18b06990167cb5e1d370a49bf |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-17d1b4b6d9bc894fefa3234a513cd2c01a0c22ded8ca0bc497abbf0a328a5fbe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d4630782aa95055d66684e6b777177bf64ca678da9283ee81eb070d1c836516f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-11dd01ab | sha256-72335612fa5dd597ba9d8b2f86598d9b0c752a5314e993c9e413486bfb1729cc |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-11dd01ab | sha256-9eec52e27b8b7f3d116087787df60d201d773550ce2b99dac0015b26575b4cf0 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-11dd01ab | sha256-72335612fa5dd597ba9d8b2f86598d9b0c752a5314e993c9e413486bfb1729cc |
