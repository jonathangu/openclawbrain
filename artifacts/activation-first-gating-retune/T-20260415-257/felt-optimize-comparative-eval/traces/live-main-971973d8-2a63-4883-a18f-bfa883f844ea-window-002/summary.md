# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2684bc9ce52da3283e7e269a65aeadfa9bb4bda12e0a5937bb82b4e7e3f59ace`
- fixture hash: `sha256-4296b198ad4b2382e867baff61985bd607aae4ddc54e4c60ef5ccb597fc35e68`
- score hash: `sha256-9537484ab68e63bd339a6364341273e7d4c41672776f797eec01534ff5e303fc`
- bundle hash: `sha256-3d62430c8a7deddc3fad2d7c314dd1a96c66a15e49fa9ae74e950207899e0081`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d2479b1210d374fe06946cec83ff362b307da973ce6e0c46c380449deb18879 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-fff96e0c27ad3ce0bbd7ed1f63f70a37a75608e67328e0e15727f010e4ffc0ae |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-0e22a317e283c7b1373e7ecb3513300f9a5353934e2508df19a888ac14ece49d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d466ccaab94252da44c729fecbc1330f346bdfbe9324325473be564fbe57949d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e0a61100 | sha256-a4dc3491f33596fc6809b1fad34cb9cec0097e556b4ec87dbb45e7a10a19222e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e0a61100 | sha256-ad61a84079020000acf0ca1a8849f866382d29edac633fe3b87a39f5d85fcae6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e0a61100 | sha256-c687feeedb1eac6d455d9bef433b3256185e23c5ba09f20eaf067f2aade296b4 |
