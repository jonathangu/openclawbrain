# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-37e08b64bbe3859c91de0e6b15ebf25a874ec18a02a748e2f865e32312052a06`
- bundle hash: `sha256-03657a7d1de0ee1b75f0e0c188a166772278a75e5e4a8a023b00ed0120dad5ce`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e54eea5dd476d45e5e7ab52a9b0ed2c646fc990677d2858d9966f3baecd8936 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-7c69f5748110853378a1eb7b750567bc1db9927f38b75ab5055da2238a860096 |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a246c842eab96feaf0fc3f42c43efaaceda6fb2633a208cfcaf146d3e48171d |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-b2c646d3fb9e058c75035301d759cb07d2f9c85c270b6baf4a8f13e66ac8b4d8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-606a23bc | sha256-a97bea789d853bf24c9d616b888f6885e6c2e6495dc6ca50dbe273468da0b6e3 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-606a23bc | sha256-60c520c6fbd835167e8bc1b8251a8c1d6040af96e3d770a4d2e3485af1d4e6c7 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-b4ef6c83 | sha256-0e395308b926e01e931789a30c0202a0d2641a6b0026927f08a008340a18b2f8 |
