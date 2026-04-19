# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dfa81f4d2b00217c5c5c520178573740e8780c6997e7fbd463fe714331cc7869`
- fixture hash: `sha256-ed00dcfbed6598ace12042db40479b3199c9a2955a7a673a786b8d8fa048ed17`
- score hash: `sha256-e13ef8d6f2f16925fb8acfd1f2926a368cd40fe68dd0eaf6d32af64b88747bbe`
- bundle hash: `sha256-f400841c2046607bfdb363adc39a195132d313879a4a145f620271f748487c98`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2d5028651743004fc65c4abb7a18a3ce781f93f13bd67703dbd698c51e61ae2 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-672e9770fd5868579c6e0041b74ae3aa98b742f9f0e9144a395a076a06af5467 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-15c7c30705e728b35d7b2ca869226522b9e7f1264e44767799126f51b6bbede7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-94049975bad5601c9cac3790f20488dabdb12400cbebd19a565fa9ae9c39d370 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-0b8f87cf | sha256-3bc869e679dca19e2e0e0bfc9dca7ca12ea2d276f90ac118691b82edf65946c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-0b8f87cf | sha256-3bc869e679dca19e2e0e0bfc9dca7ca12ea2d276f90ac118691b82edf65946c4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0b8f87cf | sha256-a2f06138255da558a7f73ef0bfea9b5c05cdf5bc73c9423c9512a7ff42424792 |
