# Recorded Session Replay Proof Bundle

- trace id: `trace-train-freeze-eval`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f728c53041fe80cb5bd8e01968dcf8bb4012e0baeaebb1be20dc5d213fcde73`
- fixture hash: `sha256-222f49747a6fc48b5e1b2d503822eee5bc04db6bf6b0ff996b0290937143bc04`
- score hash: `sha256-9a4827d8c274a23658cbca826331701fc869a6fec8860baaa7eee5ebb08cea4c`
- bundle hash: `sha256-d38ed6e31fb225c58a18675718cf10e44c1c526cd1b45bad3caa45b4fc2e39cc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 1 | 0 | 1 |
| graph_prior_only | 3 | 1 | 1 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 1 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/3 | 0 | 0 | 3 | 1 | 0 | sha256-d0d7ec5bc630c9366bf082fe9e3df18c7cc3b50bc509aba5c3b112c7f54804ea |
| vector_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-939689718e8111cf40a40f93925520987e0cbb614fc0f157ca922fe84c7f05b0 |
| graph_prior_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-212c4e7dda6b227f4dcd808d1d36a7119dcaf0673f0911819dddd045e8c8c791 |
| learned_route | 3 | 3 | 3/3 | 2 | 1 | 3 | 1 | 0 | sha256-c15652fe12929cfa157c113a3963e601ff9609f82b9ae08529a32b1ac42a6ca9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-3 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-5a3a6d7d | sha256-97df62244330b9ff46cdbaa327544021b62c7f819a4f5bcee3dbb779a0c66db9 |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-5a3a6d7d | sha256-7fc7a101f19fd8e71cd3511e6e09f1d53ba416f6b9924e33ac14ff509be913d9 |
| vector_only | turn-3 | 100 | yes | 1/1 | no | no | pack-5a3a6d7d | sha256-97df62244330b9ff46cdbaa327544021b62c7f819a4f5bcee3dbb779a0c66db9 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-5a3a6d7d | sha256-97df62244330b9ff46cdbaa327544021b62c7f819a4f5bcee3dbb779a0c66db9 |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-5a3a6d7d | sha256-7fc7a101f19fd8e71cd3511e6e09f1d53ba416f6b9924e33ac14ff509be913d9 |
| graph_prior_only | turn-3 | 100 | yes | 1/1 | no | no | pack-5a3a6d7d | sha256-97df62244330b9ff46cdbaa327544021b62c7f819a4f5bcee3dbb779a0c66db9 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-5a3a6d7d | sha256-97df62244330b9ff46cdbaa327544021b62c7f819a4f5bcee3dbb779a0c66db9 |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-6d9fe5a3 | sha256-d07307904ace59a28db9cfb0e202db0c583bd188055e100bf131e198ae4c26c9 |
| learned_route | turn-3 | 100 | yes | 1/1 | yes | no | pack-6d9fe5a3 | sha256-d2f5db7c647e20eb86c4c44521da596ada1dadc82fbb5f4a5a2039d806bf3f54 |
