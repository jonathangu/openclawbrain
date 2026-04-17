# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ed82473cf8a44c6f378cb688937e33b3ea351a6801142726a4915ee5fb6d88a`
- fixture hash: `sha256-4909ff1896b085966400449aca0e9ac319b4cd9d22c11198c9e8e1d61fedcf2c`
- score hash: `sha256-f758d3301dfd08eda99369438e697ebfc54c6f68f56a7051a0498f9dcf220592`
- bundle hash: `sha256-0bf56677440c724a94d8e6f56f6cc3e8c22b43111c072080fa2a86195ea285d8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b89f268ea78eb5e49f9755003f2aa744b81e1b854e2ac1c9e8f1a95cc59955 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-82a3b8aecab3f633e8b3012511c0173034e3a1db05277cdf01e24c110ef81c1b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7d5e32579184fd1082ef7d7b23491d24c50182530bb7e405cce61c54219e4fef |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-659445afb7ff69a52c84694bc30e7f9e46f671185a4ccbb65c584ca74be9a3b5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab44fb37 | sha256-c8890c84d63efd4eb35ce147e5f08bcf39921ae16112105fce1cffb284ab6080 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab44fb37 | sha256-74099bfd82ef48542fd0ac612aca8de5e203581171947c1f68985422df58e542 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-fe6fb736 | sha256-b1def9e050d351957dadee1fa42c683da043a9a234efa8e510fd925e705f6271 |
