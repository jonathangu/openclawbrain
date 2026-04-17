# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a76e30e78a3c6628f4a84b691210152bcddf9b8fa0661b16388a6ca59daa23ac`
- fixture hash: `sha256-2f161d785d6fb80ca3ab0af035b3aa3abbed725f829a4bae1a60b67e83a88b19`
- score hash: `sha256-f9977b6e2b3a89410597b196978dddc7b11cef9298c03cb18dd34028c537c86d`
- bundle hash: `sha256-728a2ef61bdfa0fb122cc3017fc20129c53563c4def35bead348fef15b9c42ef`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37705891572c574b5f2ed2ea56d6ec8c0372961de1b290750eeabdf9bb9948c3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-09c67098d43276123cc160ac95d4288ca37e5dbdcc7b216a7f5440ab4217e5c8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-48154e191f1bb977e0480d7ef5b06b28e37f699c21b8716b7b7dcdc2cf64aa7b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a4f8abc2cf386cdf1a83adfb242ea162dbdeefe52cd20210fbdceeebf99db43a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-40b275aa | sha256-5e50e0b8beedcbe0cb5c47230f98c75405fa272ed942322b93dafb614393af99 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-40b275aa | sha256-1955a740d258ed69c6f6d7e35bc11d92f605002ed348aa38fe79bf87570fd276 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8a68eeb9 | sha256-1e8b1afcdefe9882d4d44061a5c75cccdd6b3e0bef3794027977c8f5c0bcc543 |
