# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-34a774cb3f6c8a06b7737a6a2929058386a540d4a4f6fa06d56dab519cbae33c`
- fixture hash: `sha256-38158baa488957f4efebe2494068936f86320ad50d0d4566b804a6468d20bab5`
- score hash: `sha256-88b70117fa92e2ae3df0e78b47951fb35fc12b594d9b824e5684b5be03922982`
- bundle hash: `sha256-7dd4e44cdbdecfa597635bd238b65bddab776e5537e3795b9b4fae381399e7eb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 60 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2a941461c5687ced5f6be63f00e8602b946e4d86dfa5dfb8e215a577d1b9170 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-74e9977224225e415faff1f352a631e484c4bba7482cec545ae30c6fbd2ecf77 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f1e03618e9e9d90f52d92f827f57c495be0cfe1dc069a194af6e2572667de9e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-578e1b1d3a368a7d05cc682b0efaba49e2b0fe1e995caac709390187b3f10525 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-2b52e078 | sha256-7bfe81f65835956b589b279595f18916859ee57627b0f864f7c8d974b207e25e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2b52e078 | sha256-bce145469752f042a70052a1f5bf36a56f5e24c900e0f408c3315e47eb025620 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d402728f | sha256-024765504655c6cd2c191552652bd4e05504ddf75498de53d0f4886c9b801855 |
