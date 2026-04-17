# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8eced4262f5a642239299c7c899085a7bd53ad7880d03357a10326803fe33aa8`
- fixture hash: `sha256-5aa5748a68c006cb4152d6b9766d43523c43872689382d99e9608f0fedb263a8`
- score hash: `sha256-174c816920221fbb4a92d746564480b3e85f8fc2e75daab2477dc344ad7bbb0c`
- bundle hash: `sha256-ba2b3a258b12a931c9db92482de3907f294bfc1bd0e0d982e1d14068ea91785f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86ead80920d9422dc3144931f0210740c8474d5a0351518c55316e7dfbfbffe7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c36c5aabfc0c64182d99617e043a270b0d41afc5bdb74f7ab8ffc91518259282 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c922edfad02113926c40bd1459971a924120378e1d9e197fda66114fa7b7cb42 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-68baff0f340afe3ac6c1cc473ffc680462a22244843d693c38a61225415896f1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0a6973b0 | sha256-a2938591aafcf67485974401cb96c45b1cf638ca7eafbb5dfcf63abb410035c1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0a6973b0 | sha256-c8f54d37f074f0e64c411ac083f043286858e95a0f2cf2be3ed570e15c5c7024 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-07eab1a1 | sha256-63dc798448438ba7d8e22c25533dbd581d76869ea8dab82d6078cffc18a91f83 |
