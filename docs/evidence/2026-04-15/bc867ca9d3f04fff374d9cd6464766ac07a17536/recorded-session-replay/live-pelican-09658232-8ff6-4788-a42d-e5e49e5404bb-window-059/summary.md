# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c32e2ebb8b7b9b5fb8d50fc85f34ad187e87c05bcad5baa201a1c538d7a405ab`
- fixture hash: `sha256-e23c11c000ae3f195bff5e2ea98696c33b399e18fdc28541e3c50f4b667d3e58`
- score hash: `sha256-b0f167d1e7bdecfa6a5ec9b23b5750f513006a950c1456594b64979cf5833762`
- bundle hash: `sha256-aacec9e722aacb88f27eaa899001f765e484169bc0fad1d51d9db16b3891d336`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b40ea9a22b9a22398f998c23b04d742ab7923c5900fe44bcea6dd68bb464780 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-331647e75b71b8dfe69a57d988baaedb9d5f1f0ad71bca4f59eb7d0f0fbc3159 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fcd32b177fc5e807ee6200501443e5b2aabb9af704f86490512e7aa5d64463b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-fb1dae1728d4b4a903d7881bcc07d7719f7dc8293f251396615374673bcd2ee9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b92e81e | sha256-0e5ccdc94215ea0fa6ac3e70d697e209930de6874ed9e35c94d811d2317f2851 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b92e81e | sha256-28687180caf6c024421729b52595e264456a19d8f9bd2bfb2b09aeff8161733a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9b92e81e | sha256-0e5ccdc94215ea0fa6ac3e70d697e209930de6874ed9e35c94d811d2317f2851 |
