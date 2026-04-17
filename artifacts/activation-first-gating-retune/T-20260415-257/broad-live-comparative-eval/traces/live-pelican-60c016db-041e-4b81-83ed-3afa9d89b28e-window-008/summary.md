# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0b1bca8dd8d311ca0f474a7d9deb1193514002f9ff0a549efbdfe8a579f7a8a7`
- fixture hash: `sha256-693be8683846991e932bfa4a0d12773f4fe199b9445b669c78493c22255f8959`
- score hash: `sha256-51185587c804f70004dfc63e58ad0799b4fe785c3c9b7011728e5c7732bb12cb`
- bundle hash: `sha256-660bf1129fa3bc5ca495dffb60a76f2fffbc24560a74f35ed69404d8b8e5ae10`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a936bbb2a6bddfe389caa1010c92a0418532436fd2f50651530e961a6495d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1e87321d7c377015d6f28825d231ed3920c88186c31ef9489c4ec2e67c21ec42 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cd72cb310076e0980187e3a7c23fde46e2dd18785d712a5ad4408a2c2f02be68 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-39edb06324e391188fbcd3faa32dfc42929ecf37c76f85ddc25aa46bed8da5c5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0f07c833 | sha256-f4e2f34d07b44ffce6c97bc7bfe913fdc293ee9521f681b68a4cb4f53ebb1b4e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0f07c833 | sha256-7f43f7c5d4b0b78c07d73dfab92e03fc0d562c60dec1161fc8d6a4e79f4af503 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a089d0fc | sha256-304fbc83f2ee8c3c48040bfc430e9d96f44f36088f1f2a7e7bdc0477965acdff |
