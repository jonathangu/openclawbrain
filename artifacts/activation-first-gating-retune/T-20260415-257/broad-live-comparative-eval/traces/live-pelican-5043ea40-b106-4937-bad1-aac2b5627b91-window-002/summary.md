# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-347a801443da9e3d23f8dc976f3d286dbcc3cafa0984aebf1f93ff8efbfd1773`
- fixture hash: `sha256-3e9f54e7049625692dd39972563612e44cc8adf4a2a27dc80d450c5621a5caf7`
- score hash: `sha256-3e57d370b293ddb24ea21a4de9c8e428d55dd9793b5f05bbf2345ea1cf8426cf`
- bundle hash: `sha256-eea68bee7002fb08acc67adc73751ce79f06cd8db3429b5f8d8d37e8bd2bed7a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bc058e6f191036e6bf4f3884982c6a502fc3d927441bbbd1c5d745ba4e254aee |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-869c6a7499882cf5fa5aa392a75f001b2b10ef957464a653df73985c0e60bfe8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a6e489ad2737a775f1c3fbc6c850987f49dd9475bdb188f5b8afac536b6c89cd |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-39871ff0b7673276fc27570794b4804c22d630a4cba2ed3e8cc5e58db9588ed9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9488e0c1 | sha256-afef1b535df275bb286a8d8894fa3d5b34c5504045a9aec695a581e64f0cc795 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9488e0c1 | sha256-afef1b535df275bb286a8d8894fa3d5b34c5504045a9aec695a581e64f0cc795 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-4fde5dee | sha256-9b59b4208830bff0ded690818f4c73b0683a54527e1652393ff66ed34d44745e |
