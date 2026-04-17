# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a95260c17a69374ef7a9ff20490cb415b09868b4babdf035ba541b6d82beb5bf`
- fixture hash: `sha256-ff74599acd0d3d5ad2046fb7795a787fe8fa0e70837c98ae65f89838fc9f50e9`
- score hash: `sha256-2dff8c07aeae209891bc420a81f7dffcc281be327eb2a516b9866eb1515f3992`
- bundle hash: `sha256-7f16d99972492a19eb90b95517bc7f53e30ccb0dd4d765a0249b49d55ef51e3f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30f22d5abba84d169a9ef0f72b28eb7bd4c2afa26a7910c928c371f416decf04 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e350cf379894fd7e67f7dc37035b02b5377e8b98492929ef17100b692ccd7ce0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2062641a76245660cf6e325bafbc50906b841a7570ff16a096fdcdd9bb7f0cf6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-11a4dc8e1c75f1d10fd824e65002d777e81e14a8cdfddba8ebba6def5a491399 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2004f36 | sha256-22ef1fd6c625569290e1cfc927d7f2b4d720c47f77ee910f81f53a1619f5246e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2004f36 | sha256-67fdc15eee93e05d27a41bdbc79140fd7437e2bd9f1874e59f225dcf83da667b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bed55cef | sha256-101c20b0bb8735dd85fcc68807b177fc2c6177a323ee01baf377eed82055eeb1 |
