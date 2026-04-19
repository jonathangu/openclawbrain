# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-e562bf45ece0f7c36b5fb8be942d3c12739cc5e0d6f9fe8081abde3b17757e27`
- bundle hash: `sha256-b2052820dd6856f582edfdeac1f8f47b86577927cd19b24022c773f4b9dbeaf1`

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
| vector_only | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 1 | sha256-b5f7fd0a4ee56c1d7d2776e3fd76fc1e3708c8ed239455e554ea1ba18da780db |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-13bdce929beedae5ddbe9ab34ae5c585abe9c489d41c8589696cafbffa3e686c |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-82b2654aa6e068ea780b26b0f05b8626a1e57eeef2adc9798c773934f43193b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | yes | no | pack-1ac2f347 | sha256-e20e54727000ef94874d857ebaec7041c3130a57f5331a6e1c99bc8e079387c9 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-1ac2f347 | sha256-78a44803c9219ced82ce55d3d13c2ee2eda6faee707830724fb1788be8efcec7 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-1ac2f347 | sha256-e20e54727000ef94874d857ebaec7041c3130a57f5331a6e1c99bc8e079387c9 |
