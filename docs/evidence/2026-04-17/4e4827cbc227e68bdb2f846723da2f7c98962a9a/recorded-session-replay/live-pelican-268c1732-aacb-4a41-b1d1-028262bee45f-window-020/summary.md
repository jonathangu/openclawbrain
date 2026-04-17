# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-970ba48dfa6c96d0a4965b4677af4fd629fef3cbc40e01188dbcdc91cce4557b`
- fixture hash: `sha256-be39fb4084ab4014f594ecf827b8324c7590b1b3c6ba2cabd9bff2dbd9a1798b`
- score hash: `sha256-17be8c6e741ef629dda7113ae2c76b859953c04b8dc4250eeaa2b467fa1b46f7`
- bundle hash: `sha256-8dc7c9152eb012ffba8f502f9d39be1c3d1864c3cd05fb2c8508daaf64bda911`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a8c8fb966bff98fd7248d900de12653a4c0149cb3145489937f87d5ed585d1fc |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-102abe6b64d72b04ca14842978a1a7281758b20487563fe880d2ee62d988bf95 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da084730fc46f60af4e80df99d480d047d4f373304aecf8e398de2e7d9e1529e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-dd9c2df2b85e33493c6275b778fb54d8d6d558f666b3504c9669e15006ed083f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bb88dbee | sha256-7cd6b27929a560604db2d7c206f3c633ff8e0cc9d17ec1e118f59275d35a24fb |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bb88dbee | sha256-4cb4140b4cc1c7b416226fc4d2d8598f5423b2ec6b38dd293a08c2f63bcf3591 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-570f7fbd | sha256-cc6dd87b6bde9493eb20d75ab117b66d47f60d75905b544d53ba35e0c3e336c1 |
