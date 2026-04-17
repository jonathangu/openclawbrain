# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-79532f3b0ed4010e846f65618be48c2307af13b97dd581f294dd9d5e6325f8eb`
- fixture hash: `sha256-6cb2c0584a4478c43146057da013c5b788958c7763f4cd2e66653360656b5ed8`
- score hash: `sha256-a7d2cc7edf4360bb7aa0dfb9d5587ba3510f5724057945334d6f21db2cb55bae`
- bundle hash: `sha256-8a1665b0be529e1f8ceaf074518de929fc2dcfcbfa7d6688c9900c37fec50d19`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9133ee007dc762137868e0af1d2b1845ed239e3a386c6ccc9187f6b355e2ae22 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c57c99625c45277928969c74e9be37378629806e5b27b4e0181b2b8750316e83 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b7742e0bb54caf0e5366e3bedf14edeb717c1d5d61d36d6f09a72f83bede30f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7e1c779698700b055ad0e07747e79e52b45a7c5a727cfdc4ede016655aacb543 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-563f1e9b | sha256-96760a5ddf09e487eb791337757dc2e2d58e5ba5dd2b9f03dfd9f14df50e93ff |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-563f1e9b | sha256-4db999b99d0529a57842070efb0d7643615b8b1c0a3d8d8d4feb35c86c6645d6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7e0bce3c | sha256-694e768ccc850850495df4e451f6b3e80dfd2881679b39c7be69eac7e2e60a82 |
