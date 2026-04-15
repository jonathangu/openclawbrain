# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-04fe85c8a179be229d8c68dec97a25113ffce0ef409792233e0ccc1c65106721`
- fixture hash: `sha256-66a77cd573b5398a7b3b4867686fe20ef718501f851c3ff410c457c68968fa97`
- score hash: `sha256-0954809d2b872ffff0c451e6cc8d3f41172399bda3af6ff8f27d444eebaca5be`
- bundle hash: `sha256-f4bd5ee9f9a576b9fb59bd4419543ef85cb57faea063951b842fadb5a45c6a82`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae930e12c21c7056f67f547427d9cdedef7d7970b442aa81b3fdb75182425c80 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3158bd2886eb377bdc2cf7ac11ee89be06c4b477c40c7e257973f29ed02511a4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c28ddee1ddfcf1280a8816e95000387455580f75fe978a7051b50ae37ec076e9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5e178e57737347578620a4687e9ee70672751aad98d29dd88dfecc3a80ac0a51 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ce72ce90 | sha256-8956616120974ab0cb15bcd2371a42fddb04e27f23da0a8cd5925649287410a2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ce72ce90 | sha256-27f47592d7ef7b8493e9ed61a54b080c1f5f99b61f7450ab9756bf97c5465dd5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ce72ce90 | sha256-8956616120974ab0cb15bcd2371a42fddb04e27f23da0a8cd5925649287410a2 |
