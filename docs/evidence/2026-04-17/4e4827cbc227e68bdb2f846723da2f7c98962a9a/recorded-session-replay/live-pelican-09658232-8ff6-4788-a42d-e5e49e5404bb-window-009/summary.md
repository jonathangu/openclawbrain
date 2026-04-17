# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1078b1a70bcd22daa0ead376beedaa52bfe2cf8765ec6a491cb29b47f4429da`
- fixture hash: `sha256-48416b4518f830c212c5a38183605df066ce4a1235bd3582b824c27bcab21c53`
- score hash: `sha256-6e84ced8a07e3d1b3a52d32e6725bc58946f31b0ce77a54bde350bf071c820d0`
- bundle hash: `sha256-e7d6f2fb221c21713860b1ee0d39327743acde9e8335c024e76d93e57f5c7325`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57e6cc1ff0fcf88903029010179cd9e85affa629951b704a6bd53f2a38e4810e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3688b9a760f6cfa3f874a91db594d9554dc867539d446b95263fe2308ea3da85 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fab12716f29b138c096960d5539f782c80934d1a1135b4357b791a75c6a7aaad |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-286bbba7fc4b0b137ac23fe81b215df6331667e5a41afcc88fd44f7ac988a610 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-530660c2 | sha256-1effa792bab437f0cf8fb9045bff6ad13feb992d1707c38bf1a0d8982fcccf36 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-530660c2 | sha256-1effa792bab437f0cf8fb9045bff6ad13feb992d1707c38bf1a0d8982fcccf36 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7fc0faab | sha256-20f5866d8c7434edd295ec6b050c1c843760041aaa68ab778fb08e32c6d5c96c |
