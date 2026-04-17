# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-69ce0c4e11baa36853be20e1ca688e734c8855423d37366857eb233deb6e9df0`
- fixture hash: `sha256-c3a333635db8e86be19e8bf48de8cbd13aa6939830c506cedd85267cb0e9f51f`
- score hash: `sha256-cd64d18349bf33a724d67d10cb51824ccbf84f603ba93c049b92f2ba3c3e4c7e`
- bundle hash: `sha256-b55d5603219c2e96fa1abde59387e4429e265883c77e625978c8370b4a315ad7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-653724b1c50980255f17a34150c96cf9693658619075d0cdd8b7b4b447cb2cb6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-199a5047a73b037156704f95fd58e0b25dc98ee589d46bc1b5e4873fed2fb443 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-46db33e51331f160cd1d013e73ee3a7232ca527dc4a1d8137db2e8ad8ae0c698 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b76e53632a8b0113f30ea42b2b71a55a44600696c8c2d7f1104251f5798de665 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a122b2b8 | sha256-ff34ee3ef3f5bf58c1186a9f30e4b8879e09a158bbc596a8140c78678459a465 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a122b2b8 | sha256-a3b6dc771315df29619448ceca5f7d28d100abc67fcf687fff22425d889edb12 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5168e9eb | sha256-f4d2935d936a53e8c151212a8c70744a592dd8b64889809134c74f51d4b65880 |
