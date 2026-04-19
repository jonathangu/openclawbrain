# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f0b0e6f922517c200d53829cb727fb37d8945bfa0f48cb619647397c75b1c77`
- fixture hash: `sha256-41f6d7ac9ee841cd833f6dd48ee4c826e9ee5964cecb194b203679cdfe3cc453`
- score hash: `sha256-3c05247e637dc04b4f8dd806ce4fb0de04cb6a6ccdf4d0df3ec2433efb3ac37c`
- bundle hash: `sha256-4bc7072d087d685c6aa0a96173043592818fea94eb1ec748721a5d67c3264ae9`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3be8b80f288ad1443027bbe7441fea408977a916cc24a4d025e0ce74fb942938 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-671ff775e6d2916e77d09a612b57653f2f45d4a377f304124db32ad0c791270f |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-49cdebb96dae67f3346b7071df1614373139eb7c11b70a67d10ff5eb90f52fec |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-c23abe48fe3c29826a8a3039f53074a5ed8dc931b1ce04d714286588d9208bb2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-1175250f | sha256-c4559a0900bb93080b2cb52bc1b5c8a7155429587c7768cee9249792d3c4516e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-1175250f | sha256-2bde0ed05b313df80e5916c4524966a99c24c26733a7b9e52e1320847ba72f10 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-1175250f | sha256-c4559a0900bb93080b2cb52bc1b5c8a7155429587c7768cee9249792d3c4516e |
