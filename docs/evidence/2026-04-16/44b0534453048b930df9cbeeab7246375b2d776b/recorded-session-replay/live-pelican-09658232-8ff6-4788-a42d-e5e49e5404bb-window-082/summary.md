# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bda6b3da4ef39b29be45310328eb0566a39d316769663e62675a6105dd7880f7`
- fixture hash: `sha256-e12b530a582d1487040cb7cdaf3e1255576e9298c334dbf79363d1f81080b1c8`
- score hash: `sha256-2740846fb7fbf5a41edcf9085c20373840e033fea5691c1ed4b29647e792575f`
- bundle hash: `sha256-5af223dc0d57b9ba1a3b9c382e7b92614c415f41aa97e71d50aa10d4cc6ef21b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bab067ff9232cc412579013f9b35dc498686eb53e7f83b8de58e12e80ba3c742 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-02d9393cb9535df1a46334298f7a2592d934aeabb1c3404d22cce4c49a4ab6d2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b153f1a9b058dd5991a24ead86181d3e1ed5b496fc2447f4d3e6e2944ea38af6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-afd980634e92f7e30d67e909ed8e1a056875a42d1732671da83d3e661a0e3914 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c4a1cbdd | sha256-92136a3d3507598813ae13919a72e6fcc3125b9ab499c4d4ada3149a95823e7a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c4a1cbdd | sha256-04472e2bc8c3c282d713acb7b93309b83bfcccc8fd03a577a0ee7721c6b5bd31 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b2b420fc | sha256-f47d44f276184d99d595be99c0437710a0b71e5cea71d705391074ed7eb49830 |
