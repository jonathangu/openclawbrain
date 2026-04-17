# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e44bd492f128c06a27f22e67cd820199254d2a3ed0a6ac13485df4261f57fa9b`
- fixture hash: `sha256-cffeca9e647d7d047b9dbfa0c2bd2eddc1a7b9897467d5e861f95728aa0ee6bc`
- score hash: `sha256-1401697e2686346bf64f9bacca1a04ab5fa13e59c0d6244ab328f1aee87e3236`
- bundle hash: `sha256-ec9661f108d05e2a262006a716011afe44c4390d492076d096abffd5fbe7246a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7211fa1ebf40e19b79ecf69c6d2f4cdaac759ca9e3451e680c32982ba6c5891c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1433eae58999c594a0111cc37cb61a3b313da9eb3f040ec948d5f09330e9c9fc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55216f686232ac11ded179819616f184d76bf9afa9997bb87b2c5fa9915f35d7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-08074de5c4dcf89fd733844a6f180218fe46428ec2eb0e2a6c91cef8145bea40 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53e80c2f | sha256-ab8fa2d8ef4ee94da939128e6ab99972439c4854fea63f9e55aedcfea7ddeb6f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53e80c2f | sha256-4af1fd31849287281e24a4ec63cefa7002e323afb478fb615020dc75e6436b54 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-41dcb6f6 | sha256-dc40d7f1fafc5d20adb622c6776fec2d7cee7a5999b123ab4d8a398d528f9823 |
