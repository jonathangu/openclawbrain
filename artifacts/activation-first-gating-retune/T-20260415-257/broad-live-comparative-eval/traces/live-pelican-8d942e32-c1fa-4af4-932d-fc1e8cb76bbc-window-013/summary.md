# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e8abe8bd791e7d6cf823eab880acb642edafbee61d1547309c32e0509f5a12fd`
- fixture hash: `sha256-55ffe1baff231052090ba7af248a8c8c581b0ed9688d4757d7043a08a2fcb4de`
- score hash: `sha256-0e5b36706d1d31e62274a3b2660e4755b60913890a2010a79f86a3803d3bc8c4`
- bundle hash: `sha256-936d11602fb35656e0ffc060b930ad07e769bd1bd55a55d3627691d6e5bd9ee0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1da03994a3e5454931ba1a5c62fc1691a06d32d29326ec5baedfa4f4b490d130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-599449526153878816bdca33ed8995e3117f78d36c96d3418510bab1e79d5da7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b55e7ab6c8c92b789a4171b6c68a8976b4d5120a692ce077fe0b0579fce5130a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-674a36e4e41ab84200fa77e1640dca1adffa8be45ebbdbcc0951c3946f42f125 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce51f42a | sha256-8b5d236d93e32b4df8c5fa2f3fb75b71cd91d6079bc6ba8d87ac47f9169115ff |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce51f42a | sha256-601134485748e3f2577bfe8e9b4f77eff83290e96844a755ac74da1dc0b91826 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e139dca1 | sha256-cee54fa17578d9f96589fa879bfcc1a31e76620e84e31d6d2f552421c8caba39 |
