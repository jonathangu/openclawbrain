# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae32f07fbe5a45648ccbd2d0869190b2cb3596e4fc7c3e1299ef7f3819e0b838`
- fixture hash: `sha256-b830296ad0e542a07399e1e822eb8c0691a725d5f9135851e63c87d0c1b12ee0`
- score hash: `sha256-2c188e07e2457ce58b1df25fa326cdabbb56a9fae7d1bd3ad02dbea5793ad423`
- bundle hash: `sha256-ce11b25233c45fcbb1c88015f8f53f9b773da06267608642da5913ffb54d04de`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f883a174ba2c6d9b8e46baf2069a63ced4f1f39ba1f842535f04648f9481662 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4637639c8ac0ac8b63edf328eb354f1f3f3a1146f065e931ef43b360f71836f9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bccda59d86b8ddee952e450d98a4d2aad313dc65b7ac68cae0a39f3c3820c088 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-49f8068218e1c66770cb7154efee4e221355c39456296ad0b615180795c7c502 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9d084cca | sha256-120ac42186ae59f4e1378fef42c657247ff10e717ba5561fab9f77c23fc0bdc1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9d084cca | sha256-51abeb2b26aec1c2cd0ddab17f7b86dc0057dac38b8b76ea9b0fc0af6c9bd37f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3dfdc343 | sha256-4600e95f2fd4fbbde8cf64ebdb33c5bb014df98418ec544e219540010c77cbd4 |
