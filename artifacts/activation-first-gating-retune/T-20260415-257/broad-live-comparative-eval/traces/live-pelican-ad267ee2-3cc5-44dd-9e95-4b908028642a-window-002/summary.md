# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1a576ab7fc82836d62896c5506ba892a7997f6c29eafb6387885075368088d2b`
- fixture hash: `sha256-e830bab1e1b5c601ab706b387c4f671be86f28c4ff56747b0f78265a86556170`
- score hash: `sha256-b9725c28f920d2ae8c34c69592a81dc1b7b1ee51e1b2538a5cb705a6b1b3edbb`
- bundle hash: `sha256-7e99aaa9c0e8bc964cabbd830a1d9eb17f948f2dd29090eb2cbd2f5b02ec118c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-309a76d5f65b7ffefd710af5c6f62a81606516631b55e10f450624750cad9788 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ec882fb3b8f090e4cdbbec0c5ca3770375e038947cad976a61e5ca228f79d73c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-42e9c8eed0752292627494dcc079ca4919d7789688b5e58196989fba00eceb36 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-1adb861e26aa6e586ecd9de61e698edf5a437ec705289c5e56e96eb136be2f51 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-18a083a1 | sha256-3fd1facba4fa1b7c43fd2463440918336d575cf7b64301d3b207a8f7ef4121a6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-18a083a1 | sha256-c5e3345f78c9837cc6b89ec276332108a486d1bf31d51d2e251693fbaab61f8a |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a561e702 | sha256-d379dd60cb6226ae78c32415bc8afe94f0bab595ddf49ab75928f05657aff154 |
