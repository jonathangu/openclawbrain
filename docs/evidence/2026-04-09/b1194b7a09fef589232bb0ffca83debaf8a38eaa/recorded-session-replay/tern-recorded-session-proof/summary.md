# Recorded Session Replay Proof Bundle

- trace id: `tern-recorded-session-proof`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5bc5fdeaf5c970cb29b8a2da8858f5dde637257b3ef1f2d1015b4ae096a5f09a`
- fixture hash: `sha256-be9ca6cad4bdbeaafef84b4f5f64b804b10095017314e91f5b03454346dec6ec`
- score hash: `sha256-f29f509c3325264b35db30b22add4bb8755147ab30854555c5a75cafaa9d85f8`
- bundle hash: `sha256-0265a0daea9bd81595d19a9c01428fefa808004e2758be2a3a6f7efc5a414d6c`

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
| no_brain | 2 | 0 | 0/4 | 0 | 0 | 2 | 2 | 0 | sha256-5f50e37eaccb224c1a52f95bf6f77a322195f8a2bdef00652250ad97536854f9 |
| vector_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 2 | 0 | sha256-efd769c7fc499635463b9fc1dd227143b86439905ace859a2f99c4d0d8a80ca5 |
| graph_prior_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 2 | 0 | sha256-d054eabab8db3be74a12f7655f369a312a433e2ec9039103a858a94e70411fbb |
| learned_route | 2 | 2 | 4/4 | 1 | 1 | 2 | 2 | 0 | sha256-ac119c6c2214b9f91ec7656c887894b6aad2a717b745e53fe180f04d7618550a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-alpha | 0 | no | 0/2 | no | no | none | none |
| no_brain | turn-beta | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-alpha | 100 | yes | 2/2 | no | no | pack-2dae125f | sha256-e5301ea165cb4854f5aa8d44015d6a99c14f7ccb899011803278773da08aea24 |
| vector_only | turn-beta | 100 | yes | 2/2 | no | no | pack-2dae125f | sha256-6813a3c9128c9b73de43f4a57d5eb7e7a6f6b2053c44ac04656ea1c230355101 |
| graph_prior_only | turn-alpha | 100 | yes | 2/2 | no | no | pack-2dae125f | sha256-0dcd96bb635de8492214c237f71600b36409e45cbbfc29845886926b843a34bc |
| graph_prior_only | turn-beta | 100 | yes | 2/2 | no | no | pack-2dae125f | sha256-82cec35e2735254d1501fa76e6543d4be0f9b555d5beed76540b8e9313c4503c |
| learned_route | turn-alpha | 100 | yes | 2/2 | no | yes | pack-2dae125f | sha256-e5301ea165cb4854f5aa8d44015d6a99c14f7ccb899011803278773da08aea24 |
| learned_route | turn-beta | 100 | yes | 2/2 | yes | no | pack-268b28f4 | sha256-f362ee639b4f4367c2189d42798047775ca7cb094e82adaaa901be9b885dfeb1 |
