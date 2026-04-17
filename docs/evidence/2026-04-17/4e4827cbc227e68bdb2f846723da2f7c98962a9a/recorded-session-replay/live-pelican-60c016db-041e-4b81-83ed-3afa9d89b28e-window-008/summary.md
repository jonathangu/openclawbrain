# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0b1bca8dd8d311ca0f474a7d9deb1193514002f9ff0a549efbdfe8a579f7a8a7`
- fixture hash: `sha256-693be8683846991e932bfa4a0d12773f4fe199b9445b669c78493c22255f8959`
- score hash: `sha256-ef6682a6e4207f5fa125d145bb692fd3714487fedfd9e4962d6e8378764ad18a`
- bundle hash: `sha256-9fb89686cccb2ed729617944ca5691cf1314bb56528b4c8fd823f4d054c107b7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a936bbb2a6bddfe389caa1010c92a0418532436fd2f50651530e961a6495d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0634e7460751fed41c4dbde68b3fc9bee1a8361d9afc587a4ddee20d78b6a6af |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d64f30fbab5663765b3226cfe8eade2fa558f3d44c34b64ce7ee4b530e48f7e1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-44c6efa6b3a7bcd46ef8492c63b2ff1f93e14f34da60babde48c5ead1b06f104 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-43704cce | sha256-611b16d2b9d8cffb784cf92bc48af40d07d193e046d147e032089f16a5e711ed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-43704cce | sha256-ec6c39955fdb5dd8397e99b2b2f0b4212142e9f30766a62046d4fc024529eb1f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d4f25597 | sha256-09c62245ed0cb8667918e8da1ecbfc5f128f9076c85ee80933921a4849a2af68 |
