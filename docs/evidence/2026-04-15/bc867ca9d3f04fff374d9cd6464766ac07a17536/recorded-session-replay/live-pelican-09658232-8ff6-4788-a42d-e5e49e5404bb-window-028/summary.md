# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b96b2a97bdbabf2a2491696460b8da77dc242a25c47533759e1ca69d544c781`
- fixture hash: `sha256-32449d86eb6b142eb11e1d76d43e4c37d62e87233bae5b870977e6a064fa97e1`
- score hash: `sha256-1d625ce169eb8b3e66a668c57b8ef7afa8bcf20624cca38ec01a483e36bbb608`
- bundle hash: `sha256-75bfb725e244101e714785d87919ffbca180fd4a64bb700efdcfdb53896e8f31`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b14dd3575bcaf16e76897e36504d083be01ba320a2077714c9a7749ba84f112 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-43b5942fb36bcae908a0d8ee5d5ac6e57618bdf6263c0348e7e5a90d7c695752 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cfacc12326ca3a09e5fd7d696b1ab76b3adbc8a5e26fc57a3f8919e0fe83c91 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-74df774ac7de1a8fd14efab9334df1f03c46e7581dadbc5c212363b14c6968be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1cfc9fe1 | sha256-751f98d41513d80f4b2e697114445baee1e0cfc5a02aa8a00edc5b3881393de9 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1cfc9fe1 | sha256-969fafd296f14b472fb2fe04dc29e4026015148b8eda7e616b4aa17940db49e2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1cfc9fe1 | sha256-751f98d41513d80f4b2e697114445baee1e0cfc5a02aa8a00edc5b3881393de9 |
