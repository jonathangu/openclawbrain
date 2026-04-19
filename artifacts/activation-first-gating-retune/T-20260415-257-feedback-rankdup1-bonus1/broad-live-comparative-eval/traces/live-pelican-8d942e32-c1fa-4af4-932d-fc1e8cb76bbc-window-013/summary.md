# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e8abe8bd791e7d6cf823eab880acb642edafbee61d1547309c32e0509f5a12fd`
- fixture hash: `sha256-55ffe1baff231052090ba7af248a8c8c581b0ed9688d4757d7043a08a2fcb4de`
- score hash: `sha256-e77f0f2bb98ca1b60872fc64979491ae1ae9234f4fb7432da59ffd8b4e42c33c`
- bundle hash: `sha256-bf44194f7da1b6ea9a74b088fe51a4143a02787ad0f310a359ebcc5ec2f28e59`

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
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1da03994a3e5454931ba1a5c62fc1691a06d32d29326ec5baedfa4f4b490d130 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-99274988003833f697a317637a4eff73cf91e361766b722b34c01b593961a4b9 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-c8b30478ae4e7c090dea2cc1f43ecca22c2e579f833ca717791c00929cf69fb8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-779372938cc0445bc139ee9307b1e1051dedbf904aac1b4da1e64d090310357a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d8f61546 | sha256-e342214d8ae6ba7f293ec0b12d352db5a09f0144886f8c6a01617dead56cd718 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d8f61546 | sha256-e342214d8ae6ba7f293ec0b12d352db5a09f0144886f8c6a01617dead56cd718 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d8f61546 | sha256-406e5082d8e6e431a5acc8bc91354454d7f497b730ea006f64b8eb15257cc7ca |
