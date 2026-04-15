# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cfee0e7c7d7a332b7f5a4f24cbf19b04a7b31698d12d254a92d62753684d371`
- fixture hash: `sha256-539cc58588045f4d44638a17795295875c8ed45ffa9d4d266b2c19df9a95dd7f`
- score hash: `sha256-ca1c84d52e3601395df108d168da670a89924cae366158be1f671c1072cae5b4`
- bundle hash: `sha256-161378668c2a9a12efc12f61bbd5492a9b9813ce29cadb2b371c92de3cdc1453`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32ce668e6813134ccd828363a96ac1b89f56519737480b00962b4a14175506 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5f65974aa369b25217259dec72b24d2ea41fca4fcb9ee829fd0fbb8a234910b8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c68bb57830a7a7f65333bf7568f55c4bcc5027458a230ffe61fa590e6020a329 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6fcc9885bf0c419ea428e1874fb92f0c5c291864ef9597ca2869f2883e15ed7f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3f40142 | sha256-471f098918b373f1d27be11fd43d84db0ca082ad60376464ee5803f9626cf802 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3f40142 | sha256-a31ed6f6c02c59179849b00b79fdaef06317f0d7dc96db9c4b198ceb709a1919 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c3f40142 | sha256-471f098918b373f1d27be11fd43d84db0ca082ad60376464ee5803f9626cf802 |
