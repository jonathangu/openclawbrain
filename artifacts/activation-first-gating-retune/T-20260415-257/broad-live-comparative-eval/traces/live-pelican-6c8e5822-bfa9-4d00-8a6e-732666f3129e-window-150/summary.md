# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5ab4245c094a4283a2fe623f159ada638f3a2335ec988d52906db167e4a412cf`
- fixture hash: `sha256-200539ec6ee07f9053b46fde1430980f62e83407874931f79115b5f9bd8b8337`
- score hash: `sha256-282dfe18ec6ed899aefe20ffe005336164595c3acdee0ea70eff738335da4948`
- bundle hash: `sha256-65563a99b7612dc7458cd71b1e9048346c01c30f174e9cc10436322683d31a79`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd1ac429de4ad281ade866e710d7bcaf6542300ac52809bbbdfe005490548973 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3483d93e582ff825912f5c8e3e2c622bda271d80e1c0a3a4bd1f4d2451cfedbe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-696b50715648bcc569e9e04d07b1d6131238b11248905d0083a6c26dbac30dfe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d6eea945b3a0ebb04a78733379d652af918184d3a2c9a2199f855f35ea0a2327 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-639d53cf | sha256-58460ec112ffc1fb806dc41922e10517f79a7bda3536fb0c9a6e73d974611bfe |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-639d53cf | sha256-2f18516c4ca29eb93f2d664386c325d4923a233bece67252c1f9730ba41675fd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8106d020 | sha256-7169350771b712ca5d31f458cbcaae4ca470a2d7d1aae76d8499e0b3df818d84 |
