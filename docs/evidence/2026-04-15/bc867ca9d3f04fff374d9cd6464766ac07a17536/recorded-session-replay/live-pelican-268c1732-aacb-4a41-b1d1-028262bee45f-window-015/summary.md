# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae32f07fbe5a45648ccbd2d0869190b2cb3596e4fc7c3e1299ef7f3819e0b838`
- fixture hash: `sha256-b830296ad0e542a07399e1e822eb8c0691a725d5f9135851e63c87d0c1b12ee0`
- score hash: `sha256-3a08605601d62963c9322097365a2c4a995594f6fb43a76d2ab291321cb52db8`
- bundle hash: `sha256-13a63dac28e0ee5542366d52abfcc43367806ecc55c8d36964edaebfecb21589`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fee8bd3ac82a3f00cf370688d1c128acbe2d8ee2b6c8f142683d49120d9c6215 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b79d0e230389f071b5d47b5a64885edc674be53862a86b2f4a7b5dce11b8ed6f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-136a42ead62f700101ccdea3da22584149f319c06406364b768f02283e7f1443 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7fe57402 | sha256-09fd41fa401ab54b2265c021aa887d10031b583d2bc37539c706e3a080a83f85 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7fe57402 | sha256-191d072a17498d16d035430484d26edbb55a43a95b4903d70d6f5df138ef722f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7fe57402 | sha256-09fd41fa401ab54b2265c021aa887d10031b583d2bc37539c706e3a080a83f85 |
