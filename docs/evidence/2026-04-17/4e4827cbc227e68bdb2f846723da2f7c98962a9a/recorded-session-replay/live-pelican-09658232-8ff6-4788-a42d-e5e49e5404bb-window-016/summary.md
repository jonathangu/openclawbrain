# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0e0ac143317067e59f64740cdb9f819c48d2981153767f573c0e73b22b2b7c81`
- fixture hash: `sha256-dbbac8f5cf8c52842e2689d4f90634fa33bc0bae1bc0d3bfd9ad2ad85d720253`
- score hash: `sha256-e59a970753a300f22092c65c4e145e830fccaed90aa387df83b2f5812336fdd7`
- bundle hash: `sha256-a8192a1b6de62f582c268d94a4c82a89df5cc43bddc6996a036d6bfff809e505`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-745541acfd3bce8c03c831feeecff054c455963b939319f1092513f43c7bfc25 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-a223d55ee051b6b16910ea074507900acba47c930ebeef914797771c5b52395c |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-5618e4e3b2dfba212e21d5b4bad4cc4de5fe81d7c89dee4db7d1491b5053373e |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-109e0ba54881ea00c837c94464dde0de945ce3a48b94f522fa5bb53d2e0e57b9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-ea5d8c82 | sha256-3edebc655e01bc279e31f755dddc297f01494c66cd80d41869c1cc937c4b33f6 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-ea5d8c82 | sha256-461974a9b38f51e809696411312eeca3f5f97363ccba71bdc6b9e95e0a837e8c |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-0d14caa5 | sha256-88239412c80f30813e439c227ee80e5c2a13a1e0b62a224d21f20ebb78e573fb |
