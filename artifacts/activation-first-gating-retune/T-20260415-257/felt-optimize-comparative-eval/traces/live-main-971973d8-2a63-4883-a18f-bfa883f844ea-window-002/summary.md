# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2684bc9ce52da3283e7e269a65aeadfa9bb4bda12e0a5937bb82b4e7e3f59ace`
- fixture hash: `sha256-4296b198ad4b2382e867baff61985bd607aae4ddc54e4c60ef5ccb597fc35e68`
- score hash: `sha256-465f8e3c1d8208bd02832696e7815b0b03da2a547e4eaffe84cab027283e3807`
- bundle hash: `sha256-ea0f7752081a3835091b0b81d189b03527bbb9a9a9515e4e1084af6b13491974`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d2479b1210d374fe06946cec83ff362b307da973ce6e0c46c380449deb18879 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee30423dba86c6a2685373d764ebe23d49d167f603ec922f0b9a421cca8146bc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2411d1d9808b948f4ca6d6b298e470687a2101259770ad7bcaac2dc3e6591466 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-36302aefae6babcdaed3436cacde9670f0f7525ac99b57b65d835c68529a4890 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9f47b12 | sha256-45e3b3a421c814faf2398d7126c0a9f66e825b56ecfcd55eb01916c2fca3e5cc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9f47b12 | sha256-a89f6b026931af9d98dfb94215491367b1d473b75e2be8dcebf4e37a82808ab5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4bc841d7 | sha256-d7c79fe6b22f32394a8187aa8dcab4e3cd9ef53526f7993c2e680d67c52aefaa |
