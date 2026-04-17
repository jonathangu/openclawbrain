# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-53e7f7c2a908bfa01e8a36f987e9389c06b6f1c4270256cec14da19431b1dd8e`
- fixture hash: `sha256-4dd26bce21297c56105a43961b6bacbe27d7812f2b72d27dc4b8b7698e0474b9`
- score hash: `sha256-b0ba77eb6174e59dc5054874734a8556974dc00229b2dc810a29399543d7c1fc`
- bundle hash: `sha256-8345176219c1c2a52592aa26b7cd5f863c8404c13f9cdf46510f84d05046803b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aeccda1f8aefa0b00a23d8464e4e2bbf0fb55e8c49bf77bf016cce252f0ffad2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c5a8c7c7a0c38446f58a2c9ee2f24832ae5ecdc3acd7a1e925721c4fd67c821e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-98975e62799517ef3692920245eb929fc88da39d5d050c2c5c03f5f3e4736dc4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6d0caa03b8ca8c6beb56532b0160efdea2ea5d13fb4cb6a251875d98f0a4b200 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0945c51b | sha256-c0def28c7cd8b4f6473067a7a31b784d7066813ab53d4dffbc47d480f21a4a8b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0945c51b | sha256-88aae09d59399a44d625749f6fc2d33a7409070defc76d831fa9099dfb80b453 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0e8de426 | sha256-dce6109e1964d53c313e04f69f3b00fa0a95bdb1a4d8c6bc427b0861bc2a9436 |
