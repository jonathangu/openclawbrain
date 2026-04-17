# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ea52ae43f846ec5905bcf93ba8cbe911779b358315c7c925dfcb3bb8d88c42d`
- fixture hash: `sha256-5ca126c8a28da22685d19c74b8dc7e5cb0bac37c0b916d2162c68f83275f6394`
- score hash: `sha256-e83f5296ba3e0d8a10ace8b6461cbc237bb4ddd7a677a88e28d87fc2a47c65a3`
- bundle hash: `sha256-4e33a8af505a3a3d5371314a21adcdfb03d9653bd07469dd4fe2c81ae90b1ce0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ee8e31cf8824d425cda16aa09e9361fc2028a17b7c4fcbcc21c2fa64f147edf |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b17276a1de2ae6abd5f0e4c625acf732cee29183897913ca0aa8b5c256e4b71 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3f3480ec329abda226eb7df6fec532a650c631239b9c67ef0f012be82f1196cc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ad0e840f2c2832e858fabe35cb2188df8272d6e3a440cff81a97176d62893b8f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-284ade1f | sha256-a4567c4bfddf98e0e115c9622a70bae5d2ef8efb234232f9ee8c7778b098126a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-284ade1f | sha256-8ca43a103cbe8322caef1134ebfb1f7b96a6beb78882d256b5309b07603edea7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-581b11f4 | sha256-875f89f359b41c1ef3671fa161ea97f6d3c9fc91fd22bd9f9dc00406e10cdf2e |
