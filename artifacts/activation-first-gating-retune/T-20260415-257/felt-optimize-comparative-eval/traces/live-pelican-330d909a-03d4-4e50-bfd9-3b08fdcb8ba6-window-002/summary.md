# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3188dea0835fbe3a5c0a4bde0dceb823e483b3aad858e66d490cbabd38ee5d72`
- fixture hash: `sha256-c551d2cb8b10201e8079d270837622ef96e1675624dead00decec0e3fb02a4b9`
- score hash: `sha256-e9fcbaec11b6f6b99da4d2883ce4caad001a15548539df635d029755a3947cc9`
- bundle hash: `sha256-e85167015d299ee04c3696a797aeaddc09683ae80b595eb30f8758e2514c6290`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c7e263c626ee395c7a005cdac6d8c14b4d8e92d0d3065cdc0b98a11e431231d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5276a3f254b9df47f42705ebabd198a3b93b98b4e3221c68854487d6645d47c9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-544fe4cc622ad975b76bef6c9217c7d58425da7d8ccc020726a43907bfc72444 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-72a3c02fb966a694ce5e83cc81cdaa7b17207bd66e8ce06435e416f74321eb92 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-69dab7eb | sha256-9d4531e0b3e4f9b8fd90d9fcb8f3507b13bd8fe68a86bcf007b974a31458e918 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-69dab7eb | sha256-f14f590a13fb9eb3630ea173a06aeb58ebbcbab5d8a5f8e3fba8088986378fd3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a5a07590 | sha256-2db6f15c78d3f238d5fe19fdbe4c037990ee2a398672884585fe5f4e1f56c6e4 |
