# Recorded Session Replay Proof Bundle

- trace id: `live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7dfee8569cbeebae47062b33f26bd559b44ab7e32ac7a65f3d53fcd4f9d6446`
- fixture hash: `sha256-bf2c49e43d0148934d94e443780f19f84be1befb9f46554500ee32090d69fd0f`
- score hash: `sha256-563f1db06c250f66b6dda5e20d722b5c7a3c3a036729dd42dacc8140ffacba21`
- bundle hash: `sha256-3a01556ed560c61c7c4c18e7ee97dbedaf1fef426b114095edac3d59bded575f`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aefc644f475f6e64faeecc10e1bad33424cc557b74533b3b9b16e76adc362925 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-342788bb6ebd6872324d50d3b3ca00c592aba748da450c4591b3056c2f630926 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-1e15d4873cd225638f3431e416c02ac1d4b7dc613202fb4c6ebbdba7332047e2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f68ada74436e050c109946af557f58310bcb34cd6c69f10fabd3d68b0eede0b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-a10d4562 | sha256-5f866b13872e2f381f964f3fa7e716b07ec9b44b07715808caec70e433237250 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-a10d4562 | sha256-9a784e0bae5b587cd9c31a0cd564860457eee17711d48f6c26df6beb3ac13afd |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a10d4562 | sha256-e6263a979e680858027bcac9e0cc09055b62784714ed557be22b38a8d67f23e7 |
