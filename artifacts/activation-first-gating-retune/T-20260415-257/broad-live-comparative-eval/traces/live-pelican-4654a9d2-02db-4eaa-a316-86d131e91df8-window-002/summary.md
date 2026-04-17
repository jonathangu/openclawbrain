# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0107560e9fd434b7938c996a94e09516e9330df1381928365035d337054775c9`
- fixture hash: `sha256-ccf24038ed94c209310a49ea52fc2105449214d461e66d2dc1493bec54050346`
- score hash: `sha256-d0504ee91bf4e54e434829815b43ad3bf67edc438959f7049706ab656eaf82f5`
- bundle hash: `sha256-c73b7d694c0a3bd0e356e7ed9275fb1445a3335d3bffd4affb691db8a5406d54`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33cd39ad77bdbc78ea0e62a163e0f69b70fc53f35c07ff18076ffb99dd86c22c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe6af6ec17280c01290bdab0b1d66b3fd25ffc8b174666c40ef832aed683e577 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9e0cc7a869263914de55330b82d53c4cd72f87a8498f0abc212047eb8ca5c5aa |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6a541891da8a35fb779dcea49c2a538c12ae85bb228c164adeec33d4c5f8cec0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c4e88f7b | sha256-d969ecf3377cf47f1b81fe5c8f7173ccddfc646beb6bced434d50a565787c07a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c4e88f7b | sha256-4641027db36a877513b9981a13d492c8e58c76c31205056e6e36dc7be1a4527e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-05a6bd0e | sha256-a79f8e9cf8cf4647dae8ab56196c31a7e1a1c82da880aa9323d2aec21f33cff4 |
