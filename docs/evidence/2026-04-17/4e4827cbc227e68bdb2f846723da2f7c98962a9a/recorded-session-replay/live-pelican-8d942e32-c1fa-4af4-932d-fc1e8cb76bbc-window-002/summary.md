# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-64a26e8f5e6980d9246d302acab2d34c4e18ddd7be07096a6ca889aa90e2228a`
- fixture hash: `sha256-833c77206f16af416cd188d9e8ee18c5e59708b98a4500bfd6d7d22e62fa078a`
- score hash: `sha256-471765a6241b20bc7f48bcf46182265b710f50bf9d8c5e8f339ed74c232886d5`
- bundle hash: `sha256-bdcd1ef6157f6bf0602e21a38e2253989addbe1047b530440a80d841e4afddbd`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-947e512b0a82ec6d517ce602229a8e508d29ae58b836a4631a42b14c828dead3 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9ae0264be3b9d5a1a60718141693de4dc2f4934106dafc3859df0d995112e444 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3900cd6058305686347ccf431e366800945889815a9d474d04c7675efb7f907a |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-1c0defb03a989b6351567eab804b1c4e0de409e02aba1f5c575b25b509d6c46e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5e08b612 | sha256-a1b8eb27e2ce2ce44940a32a13d690cc461ca87c752d086f3aca3dc6b395b489 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5e08b612 | sha256-20f3b3c41fea5ac413819dd7cbc53329757551f28546a782b91dcf053b79ea88 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-590501c9 | sha256-1f455e105c262725cfe27f267492d4ecd8b37d9edd1d6b2938eac1ef7b63facf |
