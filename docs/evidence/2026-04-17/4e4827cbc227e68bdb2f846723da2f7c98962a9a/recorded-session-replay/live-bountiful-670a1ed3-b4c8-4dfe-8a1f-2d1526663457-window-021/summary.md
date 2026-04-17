# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-512c430e649faf76044870db1348b61987384f6c4b42eb2624038c368ab6a4bd`
- fixture hash: `sha256-6f2c5641408f7a03798669e19a288492bcf8f6f0b8043e459e2c72b4bc2ef9f6`
- score hash: `sha256-b967658363a97f103e3a9f3819151b57d2756436f0111569afdda2deb4a17ef6`
- bundle hash: `sha256-6de2616e7e0862c57ada47c9f8f535b19aad48feb3c216ba13195703fafa20de`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fc8b4083586f10fbbdda0686c1eb4cc964fe1c89c35a3824fb52431cfb03e36 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2306e3cc1abdd640ff7ffe6a877fbb1fd95f37abe11678600c630aa02ddaeee8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae2ad9851b3bfca5fd0957c9b068473a20f64050d6d4bb9ea536fa32c6d6afcd |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-85cb72d4ee14a55dd661df9a4b283e96a76fd1d1fc2becba104ff3d2d4c19e18 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-eddca44d | sha256-50d466ec33503d0e1d27feba185d37bc8c586a9337bc4dd1e5a574bb8cc2ce61 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-eddca44d | sha256-e6c7e3ea7d44c272972f5ceab22f93a6cf8b2e5a8b5266909817141071e90818 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8a8aa068 | sha256-ddf1b4126b7044e128f3fdfdef69fd93013f9438deb4157335d450a7a4a6b490 |
