# Recorded Session Replay Proof Bundle

- trace id: `trace-train-freeze-eval`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f00d85d59f1b55fe414cc6de49912cc1d90d9b19237510d225d442cd31babf3c`
- fixture hash: `sha256-fc23fc34b4bdcb1de856768a3fefe077edddd9b47478dcbbfc5468ed671b1b01`
- score hash: `sha256-53215c77d6eb87438165f1f84f129c9042bbef9b95124bfeda66708befbf5520`
- bundle hash: `sha256-fe2a86b3460fcc5a42890dd38114c91502bc03b86ea999ff673bf720a89f635c`

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
| learned_route | 3 | 3 | 3/3 | 2 | 1 | 3 | 1 | 0 | sha256-ef6ac22cbe8c0097bb1d80ae24ebe3fe6bf84f1a1c9d50a700153bbe14af1b05 |

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
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-ba9ed008 | sha256-7f8a5705ec5b73df8c872edf4d02a01644bd665bdbb203b38e3f7f75e5292ad4 |
| learned_route | turn-3 | 100 | yes | 1/1 | yes | no | pack-ba9ed008 | sha256-70653e7c7a2722a30e51c737876e0e4d48356d03a085394a404737f7b9864091 |
