# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cd17705850f5fd87f770e4757922f483be90c3dcc5bfff44d696c49e62560cb7`
- fixture hash: `sha256-743937076adce554085fa9dd3236567f573df76180477a11d06a07f43c4044bc`
- score hash: `sha256-dc758f26b85290b9e4a88a8ab37283570fe8906402254eca0e36aacf6986d7be`
- bundle hash: `sha256-2a3a19e105a296eba429cae6510e93cb81dd90fc1bd8fc0867d57560d03ac0d4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38ffbd4329a21a765f40f1a44ad7d1cc0603504c91e4e697e7b573151d0b2478 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-459ab0961f23c0799dd77843e20ed61ac56c8aa16ac8dae74e42003ce8dc47a1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fd4071063d88868d9e957099a67f68c3e59fb7dc54eba729bb7e8516b3ae2258 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-185dbab0a9ed11bb553c1e52bbe8d9d20bf37e28de5c8a63b43c76d323b00a89 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-15af40ea | sha256-b0013c7721f6cf81404bfd287b9320f379cd0e5120e6b924d217c965872ef44e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-15af40ea | sha256-a39f18b40e6897ce5eafe4031d6db38290e70bbd65bde8034077e835e9f0e9c5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-15af40ea | sha256-b0013c7721f6cf81404bfd287b9320f379cd0e5120e6b924d217c965872ef44e |
