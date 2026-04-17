# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3ae37b035b14582a9db80eca92cf9ea284e1a083da7e40c15c27766593c501ab`
- fixture hash: `sha256-c1a05a74d2fece7febaade02118c0528463204c7c70c4fc0e050990958f60a91`
- score hash: `sha256-e23928b4bd3568cbef76cdc71427a12bf888066ec178a962c82f54d4e0d4a3dd`
- bundle hash: `sha256-74389ef449254f2e6ec872ff141e15321df254b71199effeb915b688eb68784e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5bfccd81fb07da2148e74d03332b298ae8343e32f9c89c9de3c815764af2fb42 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e9940d560a6b1416449eed929921c4667553d789f28bec598ff1efe58d63c12e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6a8bdc66c57f11a245c05478e544e9ecb2cb840e8b383f50d979e5c1a2eee4b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-258b57dc9e58b957ea6da0ddbcb1286ec480c5d8031cddef23c89af1fedd08e5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d9643f39 | sha256-91baa2afd913c37c9010d428320835becf18907ae4800c569a44d1312562b728 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d9643f39 | sha256-6bdcbdb4e9dd612b5761e3d6a8def6b882c2c6ce24e2af60251605eba158b09a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-91bf6f26 | sha256-640ec76a055f0d57c0745784665a070d347a7b51b53c05c58ccd6bc0abeaddc7 |
