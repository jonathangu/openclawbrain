# Recorded Session Replay Proof Bundle

- trace id: `live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0f24d3812e038d9d2b67d9309de9db96cd24c2faefb0b5dd93caf569b3c1d1f`
- fixture hash: `sha256-6b6c634b067ee2b84c6981ae8fc0d6c41efb6194e1723d88dd7d0087036cd1ac`
- score hash: `sha256-78b3106ce438a2e0485d4d2a40ad10e75743b3cbc208b983653c52f0e68452a1`
- bundle hash: `sha256-1d5bc1c50218c07317d287f87303a83c2e2cc9072a22bc138cd90989783edfdb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0ea26975a8365e08832501930a2890706222216fe363c833adbd0065a774a3f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bccd8df3dfb5521867eac6273033720bb3519b617a112f28ffe8977caf0bc4b9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9898a3f7a30afdf78d4ee41295cc53aa85041b2aae35081b359bd6d11f7e876a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6d77d7a6b550bbeaa781bec4155c263d855e5a160d7cb801b0037a08c91c83b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8a6ed9e1 | sha256-dad2ad51893f8970a8077eed2a36166ab720cda27f860dbc540a0e8ce76d3792 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8a6ed9e1 | sha256-d5c327047813c236e93b8c931e4c01800f50fffc5969d27ce259d4eb149e5574 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8b02b57c | sha256-9298f2bb0c7303e300eaf83cd53809038d154a29aab6c2fd61237876b5966084 |
