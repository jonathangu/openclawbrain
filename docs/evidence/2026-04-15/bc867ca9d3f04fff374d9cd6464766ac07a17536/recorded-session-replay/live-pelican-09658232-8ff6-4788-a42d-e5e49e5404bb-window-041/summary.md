# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-76273bf572d2d6df7f5708306c7a325e0d7fc022256a8c48664d2d2c99f93d6b`
- fixture hash: `sha256-5c64fa2fb4319875db5a6403e087c56d2c1a468e1ff9a819a4e71ac1b0668ff8`
- score hash: `sha256-94ac1ca3cfa5bb381e884aad5933d18a0ecedafa4824cd740dc3626e5f1f3fdb`
- bundle hash: `sha256-3f027c00014755c94a76fa63cf8e6a7e82c8d655d29489608691b2999b72e4e0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd39b973453e541375d3824b8f2b46f3993347e9ee385b937ee54648b5838113 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-61b0fa9d2481730118de1a4593dc7e042a837bb9ada9628202a420988b411aa4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-031685fc66a0ed6c4db3cd4063c78d3d36aa2bb92201a9807789eeab0c03ba90 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6c0ea398e6827b93f68b98f2c962591d42409cb963836f5cc3df05e2644089c3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dfad5461 | sha256-c885073e4d30846da3a470c5a7bc7dc41229169126b2cf9f1e3c8f0878e340ed |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dfad5461 | sha256-c885073e4d30846da3a470c5a7bc7dc41229169126b2cf9f1e3c8f0878e340ed |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-dfad5461 | sha256-c885073e4d30846da3a470c5a7bc7dc41229169126b2cf9f1e3c8f0878e340ed |
