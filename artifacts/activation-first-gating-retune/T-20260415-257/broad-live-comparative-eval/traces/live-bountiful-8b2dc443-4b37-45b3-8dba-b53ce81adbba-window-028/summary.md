# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e5cecfcda3863d55354a9b67074a3c6ce69c277ae2b3137a3f72bd0dc80700f`
- fixture hash: `sha256-6958fe867e36da1beab1df863be77bc3ca8278fa4e3d5aeb7c88307e08cb7f39`
- score hash: `sha256-b88cd7a98fd12015d3f18069619eea8caba775b281850200f8bafef5d6383e52`
- bundle hash: `sha256-c9c2e16f8283160779afb521d5490fffeea8ecf8b36c4f4e845820b4c5f8346b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7381932aea4d1bd30c10ae36d19326006a8cb4cb3b6e5b2b2ae6dadf03b6d135 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6d02283b58b69565ff742e8d22eb2653d23e28fc4138455de3e436646ba2075e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80c5ee7166a11987ecbc1a62718c543e3c3bd98c7b196d2189467ebe14e5c1e4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-14a54832223f94ecc33f1b491028fac23f8d3edf440447ba74615d9ad09ac33c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-86721819 | sha256-baf9e0fa5e7459c9b4599ba59e452e9842c11934993f9104427069ae402a0b2b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-86721819 | sha256-71541c9a656f042406cef54438e23f5a2aadfd3fe1f4f309d069a771193684c9 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7c51c13c | sha256-481b06bb2e214f2e070d2738a8974ecbcdd898c61e6a235738b54b503dd68c10 |
