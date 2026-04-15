# Recorded Session Replay Proof Bundle

- trace id: `trace-seed-carry-forward-eval-dedup`
- winner mode: `learned_route`
- trace hash: `sha256-afe79e684511872cd52b734532e20cfb688f02de0364132f171ad94e390921db`
- fixture hash: `sha256-884b77f64eaa55d9f2a409b7b1218f0bf4d168526afc66b534b49f49d5f4b795`
- score hash: `sha256-bf14b161e80c956a6e83c34dabcd099b025c084700d613ba1eb00da315053c4b`
- bundle hash: `sha256-e192687dbafedc8e0f1d7b61d0bfba9f4849ba7ddd422d72d4e7a8eea0bfd7aa`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | graph_prior_only | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 8/16
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-d3ebda202ab0b33e75a0a4712c3568a0bbd9e431cce03b3d55dc6620c9a7a753 |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-23c3987b4e7db16b749dd62da06876773acd7b046967f3394054722df81f566c |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-224faf77df02701ce5dc2c8c04808b8d185f98ba6e38a8348a93830bf45eac77 |
| learned_route | 3 | 3 | 4/4 | 2 | 2 | 3 | 2 | 0 | sha256-cbb78b64f911578d3d0a4c78e90e512b3e54c688b86ac7635477d25a53c344b2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | story-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | story-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | story-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | story-turn-1 | 100 | yes | 1/1 | no | no | pack-5f6f2c8c | sha256-07509eecc92a00db163adc2b3c1686ff51c6977a28f473d7d003e08ed40e1055 |
| vector_only | story-turn-2 | 100 | yes | 1/1 | no | no | pack-5f6f2c8c | sha256-67e53edcfd20b897af799fc928c0c639256e09450e23f00467d6278370852174 |
| vector_only | story-turn-3 | 40 | yes | 0/2 | no | no | pack-5f6f2c8c | sha256-07509eecc92a00db163adc2b3c1686ff51c6977a28f473d7d003e08ed40e1055 |
| graph_prior_only | story-turn-1 | 100 | yes | 1/1 | no | no | pack-5f6f2c8c | sha256-07509eecc92a00db163adc2b3c1686ff51c6977a28f473d7d003e08ed40e1055 |
| graph_prior_only | story-turn-2 | 100 | yes | 1/1 | no | no | pack-5f6f2c8c | sha256-67e53edcfd20b897af799fc928c0c639256e09450e23f00467d6278370852174 |
| graph_prior_only | story-turn-3 | 40 | yes | 0/2 | no | no | pack-5f6f2c8c | sha256-07509eecc92a00db163adc2b3c1686ff51c6977a28f473d7d003e08ed40e1055 |
| learned_route | story-turn-1 | 100 | yes | 1/1 | no | yes | pack-5f6f2c8c | sha256-07509eecc92a00db163adc2b3c1686ff51c6977a28f473d7d003e08ed40e1055 |
| learned_route | story-turn-2 | 100 | yes | 1/1 | yes | yes | pack-f4cee1d8 | sha256-41d36af14553eafe1505b700753cf1844b4fcdd66eb52e3cc6ef4e23d53c58c8 |
| learned_route | story-turn-3 | 100 | yes | 2/2 | yes | no | pack-ccac1566 | sha256-0e4ec898816e7de108c51bfdae2027a11ad2fb0b103b3b29f5a9c7b1cbe03f11 |
