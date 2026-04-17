# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175`
- winner mode: `graph_prior_only`
- trace hash: `sha256-653e1762b7192e93df1dc01ad3fa2126f6513bad2e3d5a89891f193ded446910`
- fixture hash: `sha256-4262eb1c667bd83d27b33dceb3d4d1a1c6a1b57d1ba763770502ff6e7c8a4239`
- score hash: `sha256-f7164628594d63741ac8d95b19a8032d00d4ed05f1dab20cab93b4bc5ba88792`
- bundle hash: `sha256-503d66260bb1cf81badb07062d86ff380959cc90e7b4e823e5c9a6d93b8c2d40`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f90cb77dd3ff507938d3ef155b0e74e6914215ea5bf7fbd610cc02d8404add3 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e0e70f003b79ff2a3862787519d18f88ecf8e599022243b798c2a5dff9f84bf4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-06d9b405a2045fa77256854c62c458e7aa5939f753647424b68b6f497e9e09be |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a53716ce7fccfced7e5722a9efa321794bf061ba1e39f723609e7993ecd53c50 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5521287f | sha256-85c88531437189c4beb71a74c2f64cf90c478d4b1e83af4bbf15256b746f7048 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5521287f | sha256-d4659829c25ad127d1d27ebb969d2635c8af14345fe41b9eaa519ec68520d21b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-2f4aa734 | sha256-5f57ae322e9589f51c46a76f6bab97be9f49954a3bf6ac51df98bfb1134fa96e |
