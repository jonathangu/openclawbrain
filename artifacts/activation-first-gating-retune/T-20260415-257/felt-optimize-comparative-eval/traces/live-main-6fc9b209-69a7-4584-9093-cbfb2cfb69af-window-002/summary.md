# Recorded Session Replay Proof Bundle

- trace id: `live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1238bb817085e52d5386a747baa6ea8bf61e3a37516af898c3b116b0246d9843`
- fixture hash: `sha256-edd92cf0e628f6e0582722d507204fe8af0abb5e8a70f6ed2001e47aa93a6a45`
- score hash: `sha256-01d5806276cd18f3c0343a8d4a2ccc27954ffe74fc5c49ed0a9510abd18fdcf4`
- bundle hash: `sha256-b2034aaa3ad7eecff366b4f1bcf4faceea3b9a2b15c630e535a247d4a1a57c16`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2f9bcbabb6e41c0be690a68df09ebb71d4f854521659c85e60ae6817b1b9042 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-26c2120b10b0a3708caf4c180f87b87c87bf3e10c9511feec96860c131a093b8 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-e594a9cc607a08e7b836d025a17202055ebc15cfcba987e14bf86ae6d94891a7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-790ffaee0da87994e724f3b389c88cd0ec0a4196b36b4f4d66023d933f163309 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3c2edb15 | sha256-635435216f1a0c05ef3d53353c3e09ab556fff79ad9c80301566f80ce6f129f2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3c2edb15 | sha256-7052e5b74c3d7df06b57dd83cea90148c9ee0e3f244c684438e634b139cb47f5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3c2edb15 | sha256-f249f7b7402463ada359757d9618b30adbba08a7e6b08e1fca57729f3620e2cd |
