# Recorded Session Replay Proof Bundle

- trace id: `live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5348a011d171022e0b0662292622dd790b7dccb6110063ebe79c7f32c96cfe4b`
- fixture hash: `sha256-076839ee1f2768f5fb0e1a395f80dc28e7868b4aab96a489d4fbcd347a8fc395`
- score hash: `sha256-3309703224173cb1c798c88292037524b5d52ff17c486635bf873608f3689b7a`
- bundle hash: `sha256-45726b27eea78e586b8ea45a3435e691a718c03f009024a5399b9cb9bf588a70`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8419d870c8d46cacb719bcfe273ed17a6c99411785d19af335a1112c07626e3 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ba1b8d05a330be8e9384bf6135d0d6bb975e3e339083a3a9eb43926f46eb1e2e |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-0225dfb7f7783bc8aeede43c6768d4aec2b82097552f5f17eee7907383b1389e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9f57c758 | sha256-86abc5a3c5406d80ef756f1fcc40e1313c6e08675f0c620f1fb910d1e5c6060e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9f57c758 | sha256-3a6dbb76d4a03fda801e749b48f951d712c87697939b3bd7cbe6075029f15c8d |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-9f57c758 | sha256-86abc5a3c5406d80ef756f1fcc40e1313c6e08675f0c620f1fb910d1e5c6060e |
