# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7206817fbe9864fa741e2aac4263783623734861273e6c92294a7e71e4bda31f`
- fixture hash: `sha256-3fcf85ac262f6dca9a6b48603643e7ca5bbe3663229b7fc7238b9b7fb3303591`
- score hash: `sha256-a785d699362e97f93580622fa458647f3ca71d90682b523b2be481e4e43b65d7`
- bundle hash: `sha256-a8ace80115a662a96555b0a589a20c72ad697242553081a24c4fdc0b01f57dce`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a6acf0ea4807b1384f37283996c98fb6c5d3cf32e52bbe94b1a201a85fdc539 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1618f116c4a9def12f4dfbb5f4bcc1d8054ee9556510f7bfaab727064654991f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67caf35e5002dc01a068bfeb0c11d6f7fee19e08ff2af3329d28de6c23ca7c70 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-39e323a02c85e91cb1fb077fe5945fde542cc903cb5b985cc7bc1d8f6294e906 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-832eac8d | sha256-242306565e8e1d93af408d1871e14650d17d2268b740ee64d174acbf74620658 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-832eac8d | sha256-a8951eb0a80453250f5e01824327b3fd67e3f17cf8a628a872a9ced9f22e8edf |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a7c3b706 | sha256-ae1ae9cb1ede122044ecaa953663c6155c785dfedab369fd8cda491219e6dd18 |
