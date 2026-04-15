# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1daacbdd680cf4033ed5d9fa2efa105e6544ffe1129c7600ff85b76c0c2f8393`
- fixture hash: `sha256-ef7e749ef838de36d236aa29e0590a88a86c6b42be12cb84bf00123ad9c263a6`
- score hash: `sha256-7594eb3f0b00d21319823a3d655ca78346e8cd541e60001201c1261c8d2d10fa`
- bundle hash: `sha256-add29e7372504b11930629389e2a4023408f168d279291cef6dc0a3f3a496756`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-715391ab03706266e4dd92a9d6ff099345f003fb7379bf779cf731b9d18a7950 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-67cb2eebdcb1df54f12e69d0cb217dd9253fc8b4000c69ce9eae0d15cc75a304 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e12675517b610eb6c4a5baa23d811fdeb6e8ef86c21355694f68c4b6389a3b48 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-eb7de9de6be05d9aad42b44862f3eb8335a69260a3b35932a9e5c30843a73085 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0b998b23 | sha256-0bcbf38aedeae61ea5524b6c9366efac652251fc995a8046d52c615476e38ece |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0b998b23 | sha256-4727fbe8bdd5956dbd30fcb7f9c0f0f3932915c3642801532bbb7bcd0f0fbfdc |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0b998b23 | sha256-0bcbf38aedeae61ea5524b6c9366efac652251fc995a8046d52c615476e38ece |
