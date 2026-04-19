# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-255cb828cfc3b54a6910bd2ccddaffb0dd5a77237579bca151a6d4ac10b7f7ef`
- fixture hash: `sha256-6bc9cf75cae8800470997ef07fe6de902accade663aac8d00dbcb4e5cbc81f77`
- score hash: `sha256-48790daab17d20b8b2ba8e36f2573621d319af9a3bde51f93e0af8a462dc9560`
- bundle hash: `sha256-94442023edef7bdc751b9af5ecf3be75f45f50fd53bebdda3f7560a386a357f6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-297b1309d9c33c67d42bc2f9d113a9c7958ac8f0888f1f62541fe7db873a4e3d |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-94a95d63f75d4f14d473cc6994ee10ad2cd5dff24a7b2bd1519bfdcd882cd1e4 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-1ca939b13cc2e52c7734f8bd4342fa5664aed6ee086ac243226437b1bdaa82f4 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b2489ae123ed08cca358299665cc03e091ae56501798e9c4db307e510fb115e7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-1b3a05c4 | sha256-2313d964790ba14f7455ea40f1170e59757ee9cb0a53201547c5798ce66bb5ed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-1b3a05c4 | sha256-730adaf45f4cec435c539673b7e25d912317c2d63d1134fb9fe81eecba35af70 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1b3a05c4 | sha256-2313d964790ba14f7455ea40f1170e59757ee9cb0a53201547c5798ce66bb5ed |
