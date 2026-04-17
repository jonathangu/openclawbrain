# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175`
- winner mode: `graph_prior_only`
- trace hash: `sha256-653e1762b7192e93df1dc01ad3fa2126f6513bad2e3d5a89891f193ded446910`
- fixture hash: `sha256-4262eb1c667bd83d27b33dceb3d4d1a1c6a1b57d1ba763770502ff6e7c8a4239`
- score hash: `sha256-3decf5e2a83fbb8f562eca3baabc4a6d96d39f3234808debd6efa4d160a05fb6`
- bundle hash: `sha256-6dcd17afc69263c14278805b14a3439c93497837b6735ddc6b0a2e19a35587fc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f90cb77dd3ff507938d3ef155b0e74e6914215ea5bf7fbd610cc02d8404add3 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-302233ee9b3c215a6735aa259ba56241b895f0f22cfb9748054d41a3f0737682 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5bc62c1ee2d289cea22a4e2da4e62966a83a798fcd41af535521a54f1000f35e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7bd5e012cc5e30379998eede748a94ee9d1fc169e5e1bca6195eaa271969efea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-faf3bbbd | sha256-7289c796e85cef2c7ebea30aeec8a388296170398c7bcb247329a23b857e1f1f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-faf3bbbd | sha256-f1dbbfddbce6ec73a40afe89543a43b37700e254ae61c581e5f461c9c209401b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d51d3a72 | sha256-4ced3f64d13bc7c3fd7c24ec797c3425af0e86683b5c3f094e2aecc90cc236b3 |
