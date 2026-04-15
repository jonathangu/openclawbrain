# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d0b5d294f5bcac07c81e1e9b7fbd08fc02ac60b4a0afd2bd2ad3564216748c02`
- fixture hash: `sha256-10c2797f7098132dfd19e74471fe861e4fd990acaf92ba667dc395a281a0c32a`
- score hash: `sha256-44a4289b482b916f5529c8ced558395b6f265f5728385858e8cfb61e7987ae36`
- bundle hash: `sha256-6b007fefee09b190a72c307377887a5eb10d55aa760da66c86434891c11b695d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb29fb5cb8fb01ea6e12d04715c0ac66ad31c35de2501ab2ab9a23569a1d387a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-712b24e9783be2b4b7a4aa3629942693d2766d93dbcd67edcab5ae68e244e7d4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-52dc2740b5838ca6d4be5d5c6f1d1a59384491c4efc24d8f1f3c9832cf73fed4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3968b85718bdb84f9ad8ab7290a240bae12179bcc97d3a0e597e18738438dfff |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-97a036e6 | sha256-290ce1c529a1c2cc5eb255993fcc8ebe42a3c98de62443e948566b001cd2b207 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-97a036e6 | sha256-cc42bc0e6ed01dd04ed33ad40efd5306cb185f712d8b96c24768c10b8fe7c202 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-97a036e6 | sha256-290ce1c529a1c2cc5eb255993fcc8ebe42a3c98de62443e948566b001cd2b207 |
