# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d8a6a685e755dad1fa0654d4b6b59fae51d32befb5054b1345ad7441ba8df43e`
- fixture hash: `sha256-6c5d4c7687666f33983698e226b698dfe912054eada98b448401e6f4fac93956`
- score hash: `sha256-a622c22298ec5adc3f4edb70fe6a9212748ab6b7dde5b0c8d9c678e641257316`
- bundle hash: `sha256-aaed1c47b763561946980371b537c55af557f7e4d3484b5e14da65c644544540`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ffc68f5afc82e7216af198fb6c91f17aa507e3e600979abda3e7dedfc7ea0fe9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9f70a38cf8dda79f05bf8e1dd74db070497b21039f6b177c61877e1dadbc5ec4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0f2df765707b5b52b309904fee63c61ab3e047d8aebede4ff8b3179196b70542 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6b060a898506f64616c4d02293344e9e95e3e5aa380667913c56b7be502a0d2e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f16efa9 | sha256-f5fff58b7826d9276774c7906aa65167141510c0b13ea0fbc728be53478307d9 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f16efa9 | sha256-0a8535505405756a8b90e70f45af77e2c2d68a98b5397c850e273ded9912e221 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-bc0ea1ee | sha256-1ec2f902e25c4d6b6cfa79c7ec3ce7c1a2650147343407efc60c70a6e3c204b6 |
