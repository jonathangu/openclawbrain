# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adbfb582784ce9c57067bcd682b42040f9ff5a4fc2a41a6b215fa1e5e63926e2`
- fixture hash: `sha256-1b81b9ebc5b6e57a68ac36d63b63963fa7e0e03c9b05269658a97fc89e8025b0`
- score hash: `sha256-66b7bd57ad85f8dfb8b6e4aab7bc8bd67a75ee3413260d9a277de3b726dd1047`
- bundle hash: `sha256-4921d8304a2f1a7495cb38396fd611549287b80a803642093b58620b8ef83bf6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d918c89a43fc84e7a627af305c1d796a487842c9f1cf040b6474472ff6068ba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8dffa3bd84f401b04f979860feaf8c7810fb6ba2edc18a791cd20c5097ac9f14 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b4ffbecb2b96c6be6b3f4f8d9ab1d4ad0ed80527c3bc3ecdf9ccb02f0430599b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9f2e56bb15b8fd3f2ef98fc705ea1dadfdbe88394d51270516e8241e481a43c8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-666c9f3e | sha256-eb0bff83c71e9db8c1438ab998d9f19701bb4b61e6f04bcb9b842eb888ad7c90 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-666c9f3e | sha256-cae7075b33b8976917fb521353352b03c1c57eae92acccc7c44bdce36d53f661 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9d3d5a85 | sha256-df93e339ccfd34a29eb3842cc78a8a7ba7ddfff4f9a51175fab59a1be1661658 |
