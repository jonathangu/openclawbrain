# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-cc7bce9fc4f64a4882791c2b360d174a288a66e981216ebf3316f29c6f948191`
- bundle hash: `sha256-9938661101d637b7946fe65b0bd9204ee3fb1af4354175c1b0093d9949c67022`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 7/12
- phrase hit rate: 0.583333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
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
| vector_only | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 1 | sha256-d00ca89022276d979054639dcb33bb4570512791a0e402ab612f41d043d6a19d |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-19ddd211bd9cda1de92a9569223cdcc3c5a0548dad48f2beaacd5b89d2199c5f |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-e0aba334f54e68cd93658d381dbbb9ee8fd706a41744eaa801aa3657098f0347 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | yes | no | pack-d5eabf0d | sha256-0d270ff425add01912388d791e329b09e6efbcd7b1ab61e871c6ef4673111cd4 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-d5eabf0d | sha256-ebc0ac29803cbc2a6a38e0bf9a130ce7a17cd96e5a2f34d0fe9b84d53a90c8b7 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-d5eabf0d | sha256-cd9d172f1736971b64b16fff7b09e6f184b1d685f52618d8afb55fe84be1c8e9 |
