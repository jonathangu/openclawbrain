# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ebe19ca9bc459ccc52f05cb3ef8e24277b70f060cd7939f534aee99c63488ae5`
- fixture hash: `sha256-83e33e2dc3d5736fb8b475959b3f799a1522431a9ceb8bc4c7fc74edb18967c0`
- score hash: `sha256-5c7fe4bdff10dae60d05709a35c87bdf824cb69e039d80f45657155da85f5e71`
- bundle hash: `sha256-ce75d13f941bd99cc9dd9b757cd97630194c7c27d70b7a66a414ec988624da1b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38dbfef9dbd8d3a664a1f95db7f92b5a52579781d0eb6bee8aa47758b54b5ce0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cdb665e89875dc823661bb9faf99fcceee87c9e000c45b6a6327abfdabde9e0e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b6145659c1487dcbf8af2344241a0427b713be3fd0bb3e08f3eaef83fc7a260 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3236b9b54108346f594f40fe430df7e2145683be64c4de47cba99af7739c9f0b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d962700a | sha256-add24f198d1ffbc2de81c2e38d3b802b2dc90bd9a64171257170950cb036e04c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d962700a | sha256-c6bd9840528e29d7e2e27deb95961ac223a3c7a19200fe6c946cfdf8a40e4865 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-039e14cd | sha256-f2001ea8cb84751231b7bca1362cb46429992ae94c1724e1d048d37895aa9f94 |
