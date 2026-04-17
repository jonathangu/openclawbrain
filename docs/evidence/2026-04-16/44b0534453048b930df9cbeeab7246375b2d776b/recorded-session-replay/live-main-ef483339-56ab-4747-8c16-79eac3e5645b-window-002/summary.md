# Recorded Session Replay Proof Bundle

- trace id: `live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5348a011d171022e0b0662292622dd790b7dccb6110063ebe79c7f32c96cfe4b`
- fixture hash: `sha256-076839ee1f2768f5fb0e1a395f80dc28e7868b4aab96a489d4fbcd347a8fc395`
- score hash: `sha256-d423bfd77b25590090d126e26086ec4fda59c8ab0a914b2c06b16e1c27628126`
- bundle hash: `sha256-0d050fd4c1917b339a983c8dea861b001cd2623f3dd834f466097b0a62178c3b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81e33f098e21a8124ae6ec9568c1d72b0f83fe94e5b59e948eed0392a9dc9438 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec996a28b0a106c53660a89e8bdd2a618ed3a64c4342320d9307650e211df8f4 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-69d7e7815e1dce08ad1dcc99cf446c799452dfb37f2729529be8991d8713e058 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-ae3a6dc99d65fea243431fb73b0dc3d379bf1235833b340829f18a991c395765 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-93a4a306 | sha256-db87226b60bcd04541e1be3e36cf5005a4ebe1dd1d8293c56eadfe8eecfef18c |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-93a4a306 | sha256-e6c51dde2343fa080e65b2b8bf8820b345299dffaee82197ec96fe245cc3ec98 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-d2652fa7 | sha256-ee53319262d5b4e2689554b7f01cbea3895d1eb7fe5f14cf455bf886b07704eb |
