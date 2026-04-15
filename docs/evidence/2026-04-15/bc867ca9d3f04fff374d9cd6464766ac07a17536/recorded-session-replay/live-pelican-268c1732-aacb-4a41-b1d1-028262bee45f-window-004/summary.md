# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-380e12e9dd757771937f4748557c11b50a1f9a231591dd724ca65839af3ce6a8`
- fixture hash: `sha256-86ffcadee00971f5c46315d2afa19ae2e85e45bae4dad0e458c42f57f711f9d0`
- score hash: `sha256-4d051be8fbbf0f257cf82208a84796c2eae6dc24be1ab64c2652ba3e422c7151`
- bundle hash: `sha256-513503d11adcc5be0142070c41a061449eff1c2ed10b4f9f377ea4964514d118`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0928afcbb7b85c41b2a1d624e920cbdedd75575cc8baa6c3ef5218e9d291b99a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5e6e0c2784419c2587540a3f0ae9adaf1719b51dd6445c29940aeee8e6262262 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8ab9efaeecb656a5797f547ed4b6a48f2490811be5dd903e0deec9dd63a66d6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9df794fe6fe09c9f7854803624fbc8f69e7083bbba760507236d9319aec60f40 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-712f92e2 | sha256-6a1d8f6c503d3c359ee4d582901d055b9cc49278402eef16dc8884a986a4d85a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-712f92e2 | sha256-d69e9461ab314b854ce10200e49cdf61de838b8eafba87143ad2dcd2676b45ac |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-712f92e2 | sha256-6a1d8f6c503d3c359ee4d582901d055b9cc49278402eef16dc8884a986a4d85a |
