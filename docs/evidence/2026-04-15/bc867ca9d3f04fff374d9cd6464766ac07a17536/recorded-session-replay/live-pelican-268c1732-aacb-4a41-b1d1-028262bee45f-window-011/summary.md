# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c761e47d47331bf575b7d002c131402195e8ac49d688ce355a015b14825d3acc`
- fixture hash: `sha256-128e53eb7404fc5c5e08cc33f7657166db8766e76b0fe254b4c32e80c9220dde`
- score hash: `sha256-61885960a2097b50e60ba26ee94951b109e3e9563314d81aa1d9f10080d46d63`
- bundle hash: `sha256-98d396239c096a83e92e0073bd8fc1312d38b1cc09adbe7457517393c9404942`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1631644a30db1cf3d02f7c72d9e973f8085c5ed6318e74e1e83701e3e901455 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-14dc0e9e37568d588dc740fb44147162cd0841af586e5e748e346e0f817aeb7e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0d08a93ef3ad6c0ebd38d1e74f889527610f0d2616dd34ed1207541c3c19fc3d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3e03ebdbdcd8b8c5aa9ab895f6d97870a21e56ff5fc57a2ccfb40347f6f8793c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9509170d | sha256-cd91d1b8f6770700ab9e147dece58f8efe40a16a8d5c986980b8402551cdbb93 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9509170d | sha256-20cd09290531c931bc94dfdb40e32f32205855a6049436ce76fed30231b92cce |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9509170d | sha256-cd91d1b8f6770700ab9e147dece58f8efe40a16a8d5c986980b8402551cdbb93 |
