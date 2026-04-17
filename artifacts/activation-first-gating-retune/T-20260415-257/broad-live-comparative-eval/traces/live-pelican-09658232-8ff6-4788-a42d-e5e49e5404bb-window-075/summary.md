# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31f15910a7f37f6942dfc1fa59eebabc12b733c2fdbc101bb92672de7f721f0c`
- fixture hash: `sha256-d3d3b7c9daea7f5dceb8bcbc7d0b182082662e4eea5368602c8cfc65a5234e7a`
- score hash: `sha256-29b2c18ad1ac787337ef6673fb122fe6038d7b72cd6d56b29cf09525ed195772`
- bundle hash: `sha256-586704a4d10d3f7f9dc697e35fd1a901ef6594aaceb49ae97181a972885278ee`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eee7751ada66c814393120538ba88242a0ad04eb627a4b24f36524aa1be2a704 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdb09eb2d7745165b36fd4b61766c7a2dd5c5b305a2e5c88566c6a6aede6d25 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-26038aabbc0946aaa827b41afde34f466d2c5d59e45a2147b152745a48b4c0c4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-35ee90f2943012672b6d437cfce5e61514650a7f318a04b88b8c2a44457d059f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e18728e1 | sha256-07d46bd64ce8718e4f5c2cdbbccafd2891bf299153b19845cca311b160bfc1f0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e18728e1 | sha256-11db6c5bf0081702e4f23ca7366eef3df72a481760505032cf688cf17cb50e25 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fa6e8efa | sha256-e1e78ba582750067a0b3432ac6b9e8cfa3554af946d8054f18bf8a372ad5af18 |
