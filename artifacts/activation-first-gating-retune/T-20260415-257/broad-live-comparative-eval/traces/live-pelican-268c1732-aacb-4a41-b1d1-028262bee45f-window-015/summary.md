# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae32f07fbe5a45648ccbd2d0869190b2cb3596e4fc7c3e1299ef7f3819e0b838`
- fixture hash: `sha256-b830296ad0e542a07399e1e822eb8c0691a725d5f9135851e63c87d0c1b12ee0`
- score hash: `sha256-9a3dafcd4983a42af65b3347c35e02690b624ee1c1084b1032f74a49312898fb`
- bundle hash: `sha256-7988acbd6a1c9ce701a84db68f5e3eab9d2e5946582d3f134bfd3340bfb93df5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f883a174ba2c6d9b8e46baf2069a63ced4f1f39ba1f842535f04648f9481662 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f6a2b66daee430d18451e04559537b50b14662908184fdda51982f5e42aa0ec0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-797d2e6f54ddc874db276eb1f5fb2cb09c90990b4b53bf62f9a544f980e67726 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-911501914ea380ac35e141944e51a365a827eeefe011ddf5f035c1ceb6258091 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f735b98c | sha256-65f9a5581e910e8478e6ce36c73b295158f31d78f6a0a1bf68af65dd7e9554b4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f735b98c | sha256-39a64d6c08c46a7f67718a4df04763f04cab241492da97a0e1d51b4bd31397df |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-982b3005 | sha256-df94696b09d4e75e90a7b88ef796c8b9db52886c4134fce89f71755795cd05ae |
