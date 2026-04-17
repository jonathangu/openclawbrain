# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bed00a4e07440963fcb335df85735aa4cdf299e8eeeee26d6072510f9d967592`
- fixture hash: `sha256-7b9b3de84eea4b8f6489862313ae3b9c5b0de1ba49de793c86ffbf0e24eac4d6`
- score hash: `sha256-2bd5d1bf5be0150292aefe85800a6aaf303808a9bbacf6f014976b5eb4117859`
- bundle hash: `sha256-2e6482cb3e9e5a15e0fca52a6d656382fb82ac2b2858c382f7e31f3eeb01fba1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7244838abdb700b7974c4fb03ae1270d3910510dcc5c175db289b9a82a5df872 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3a70e4a7602871594b85332d54367b7db8d13b27dcfa6f3346b9c616267645ed |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-643b155f2f2b927d75263329cd1a11436a9e02d1f689a389066ae3dd49ef634b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e24e17ba49d8971d05f2154a114ddd009a160c0a3ff8d726fd18cd0129b77d6d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-55d45b24 | sha256-bcc333027bdd7840f9828cd907ab2834237c020ab82babb46aef627f0eedda68 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-55d45b24 | sha256-47dfa564ac20ebf4ac500ff819afaa62ab758de540b47d61cea4813a186d1a91 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-146b3ac1 | sha256-b291656873c85fdbee1414b64907328cfc98e2934e0030aa5973e98851e6f5c0 |
