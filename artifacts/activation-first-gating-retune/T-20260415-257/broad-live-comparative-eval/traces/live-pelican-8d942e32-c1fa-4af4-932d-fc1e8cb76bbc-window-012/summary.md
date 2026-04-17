# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-874a83098560adaa94c38c7c63cbf4c86efe4c86090d606bbfa34849e336a8c9`
- fixture hash: `sha256-b06776d862580d01d558132918aaffc22b9130c1387f99ca2438e1c6cbf7e22c`
- score hash: `sha256-a18f6bf033fb4750d9465323e6029e8eddeeea6189121904828264f2a42bcc42`
- bundle hash: `sha256-11d39f0e17bde60ba353da10a42038db06e83fce7a94926ba1c854e4a3dfa200`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74c569833e5bf27fad2f2f842fa8eaa7d60bb320f690bc493bbf6c394f309f6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-baa47dc7a34e48efa363cef12030e589639e6bbe5927d5ac769e7c69ebdf80fe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-029bfc8add59ce98a131b81faa311afb5f1b5535def3ef28916b7cc0638b41b8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-59a7c449a42efe1754b6bbd0325d0d18d1d34af4ac6300e1ba01b6ef0e583302 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3fa24cda | sha256-ef54863d25d0434e845d5dc4cc0064eb03fcb57b9cc0e77d037f763150d586ae |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3fa24cda | sha256-fc34ca1379891462c72dc4de866f7975e4e23cc135a98bb6629ff8d0d8dc37d3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9853bcdb | sha256-4e3a045a5435aab098c189ed31daaca20a59ad42ca5f1e1d091615adc583c51a |
