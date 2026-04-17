# Recorded Session Replay Proof Bundle

- trace id: `live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5348a011d171022e0b0662292622dd790b7dccb6110063ebe79c7f32c96cfe4b`
- fixture hash: `sha256-076839ee1f2768f5fb0e1a395f80dc28e7868b4aab96a489d4fbcd347a8fc395`
- score hash: `sha256-20b3a3aad1730a9a5e6006a02c6d3d675c39a0d312d9f84d63482ff6dae85007`
- bundle hash: `sha256-a8279bb36f35b24343315ca5a0788384c74f3b167bde565bc99c27d4f4f4976f`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e156d9f3bcbc6d6bdb4d40c749609308ff57ca5330326b5110e7cf5088920f5d |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-f2a8160eda950aa7149a7b74980edffa5ce73a063a54ac99dec6f3fe93d790fc |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-12134a1167e1003090162d0329194bcd59635305931107a2c896f7c35252b9a1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-6872046c | sha256-a7f2187cf9ebe5958b857addd01f5c3ab790df8f671fcaa37a171b1217af1554 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-6872046c | sha256-2c8ed3a391713c09bd8667282e54a2c63914e5c3fc5e7296cc122e4c71aab4bb |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-a732910d | sha256-36871899a376a03ed2a123a1fada6d42bf0d31eefd771e3565bbae793d93ed28 |
