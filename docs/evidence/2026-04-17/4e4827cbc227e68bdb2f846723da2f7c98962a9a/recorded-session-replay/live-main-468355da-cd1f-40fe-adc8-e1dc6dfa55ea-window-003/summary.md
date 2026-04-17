# Recorded Session Replay Proof Bundle

- trace id: `live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e484b4badd2d1a3a3d24ab18ada126ae37897ad6b6cb5ebb205f801adf4b59af`
- fixture hash: `sha256-7081875ca4f0fc3a1b3a1a20287fd5ff9fc1f2b16a465a1e2418cb78ad0e289e`
- score hash: `sha256-ec7e9a0906761892be155a14f231fbeb1c44c844999bc8ae12ef19a1fc40096d`
- bundle hash: `sha256-d50f4de9c4e2ae4c94591f8931d801fbe09deeff83b88329a27264dbc1dfc054`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2ac0d8d192c8a52c6289c0c993dfe551953686d8e0c4d297909e405aea43e25 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6863986f78ba48e126fdc271f20d099036042b462404a2fd5bd13870aadf8f0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3cc69c88b39359f8d9dd88671599d6e4ef266d783e61c836dcd4fd99e22a36c3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e86d8ba4b7090afe55ef2f9ae907270dee854ff3ed96c2db115717355f53990f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec9ab6b9 | sha256-1ebe46b33fdeb12ffec7f0a931f63130cab027c33344e5fa06e6d020fd216348 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec9ab6b9 | sha256-1ebe46b33fdeb12ffec7f0a931f63130cab027c33344e5fa06e6d020fd216348 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-e899c44e | sha256-a7da2ec103d4d1c32531506fbb8752087b0a943a986dd0c842318c56c7cf16fb |
