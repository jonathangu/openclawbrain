# Recorded Session Replay Proof Bundle

- trace id: `live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1b2b13ce0910158e65496491abf6c903c5dfb4a0455709d2493e613749674539`
- fixture hash: `sha256-f054489b05d16d5a9f9a5c47426c143ae1eefae16f2ef4a677bba49745e4b5ab`
- score hash: `sha256-c133c04b49f8db98fc6aa789a72875bd4c8447b69cc43d3378c264c5d92fcc43`
- bundle hash: `sha256-a4ba47167b8092db8d6a8aa439454f19e4d804dda4855393586468efb2c140d3`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b85312fe12721c0ca336fefffefaf7611e66d0c3fa24585f0a8f1c80b737da2b |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-b97e326d59f4b4bfc006d4d74660bf52e215a5ebf8765ff158e803412690f015 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-6256a8ab3c43b262affed2b12f2646b622ecb2c8ed3f553e1fa59198c1ccccb6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fcacb60598d801bc7ff115c59a19e83332e069cf4e29c80a9188cec207376ce4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-51be0e27 | sha256-2c0da8c77b62f27d076658f8bc0e2e79b71fb0ac6cd74b114068398c8c1e0cad |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-51be0e27 | sha256-799185a8cfbee8d2137f080a959961d58d1a2a62418303e751a15d7affdbf9cc |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-51be0e27 | sha256-2c0da8c77b62f27d076658f8bc0e2e79b71fb0ac6cd74b114068398c8c1e0cad |
