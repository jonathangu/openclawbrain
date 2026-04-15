# Recorded Session Replay Proof Bundle

- trace id: `trace-direct-answer-reproduce-eval-command`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5bdf7dae6f318a437c25599d1163e02df616c6f8d8831f793342a828e48d8f56`
- fixture hash: `sha256-9d178dbf6995a5ba8dcb2d1d1ffe4156b462ef915f606d24f8e99715f78f6ceb`
- score hash: `sha256-1d3eef9d904f9393e4377d4ee6ab694a221a3840182dfc5769b4319014606585`
- bundle hash: `sha256-a2f5cd523399986ae6d5829ee9ebcca8dbbe9070709abe8315edde9659efabf7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/16
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/4 | 0 | 0 | 2 | 1 | 0 | sha256-a2a7115fd8aa4c7bf4a719e6ce12d1ded35b497046b9e9542ec7ffd21f7790fc |
| vector_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-ef727384578aa04f416480939e7a5c329ddda19d22d4ddf713317f80d532e782 |
| graph_prior_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-1dd6b13c3852776c70bf1a161d2a08150b4a788bc4f0cee8e4c2583f5eb7c010 |
| learned_route | 2 | 2 | 4/4 | 1 | 1 | 2 | 1 | 0 | sha256-a49978a2553841a1ac9cb932d64d7a046d085e68301630bf5324b281955a40b7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | reproduce-command-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | reproduce-command-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | reproduce-command-turn-1 | 100 | yes | 2/2 | no | no | pack-eef0ddee | sha256-ca2f4ae25d6195ce1f112bef9078fc151bc7e81370eb8b1c426de21552be8af6 |
| vector_only | reproduce-command-turn-2 | 100 | yes | 2/2 | no | no | pack-eef0ddee | sha256-d3b99af7edd3c54430d0ac4f0d6f29de1fa62d80bbaa795a33c6cca65c224ce7 |
| graph_prior_only | reproduce-command-turn-1 | 100 | yes | 2/2 | no | no | pack-eef0ddee | sha256-ca2f4ae25d6195ce1f112bef9078fc151bc7e81370eb8b1c426de21552be8af6 |
| graph_prior_only | reproduce-command-turn-2 | 100 | yes | 2/2 | no | no | pack-eef0ddee | sha256-d3b99af7edd3c54430d0ac4f0d6f29de1fa62d80bbaa795a33c6cca65c224ce7 |
| learned_route | reproduce-command-turn-1 | 100 | yes | 2/2 | no | yes | pack-eef0ddee | sha256-ca2f4ae25d6195ce1f112bef9078fc151bc7e81370eb8b1c426de21552be8af6 |
| learned_route | reproduce-command-turn-2 | 100 | yes | 2/2 | yes | no | pack-04ac04b2 | sha256-e51c6a0ee6b7a420044d685da39be2c546f1bd1c7cd11450915d96369293b559 |
