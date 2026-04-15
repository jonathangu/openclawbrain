# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-823695d70f7872b1ae9eafb6d1d27250c7a30f3c8da0fb3fac149eb03366ef43`
- fixture hash: `sha256-bd84df8e56b4c53a26fb492fdea7511a22aab4ac1b787c58633c40d2b1aa4455`
- score hash: `sha256-9ae98292a02a011d51e171d07bb669696e122c82607c70eb9033b4809bc1e78b`
- bundle hash: `sha256-29add3b3ae8271ed0cac356ffea73eeeef5ea5ebbd57a766e1f4f99d4191a550`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 70 |
| 2 | vector_only | 70 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a1db4bbe90ab058f57bc7ae6a54f5aaf2daac0fc5ad242f5b0e6f3a965eb8e61 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-b0a5efc741f0a4b99f2e8ea3dcb166efa2ce44789955835edf59e400e88151cc |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6a779b2f23b9d0f9ddc136862062387798ad6b283a1ed67c2edaa7e0ce41af8c |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-19fb7e9174962a7a22ceaf29c4e5ac320e5280eb0cba40f7a01183da05b86bb7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-f691c946 | sha256-fddd8cad30e2712f948f269f5a6a6facb72e7329066a3c593cfa291c13e0a106 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f691c946 | sha256-e05ebb52c586bf3626a6e3ec0dfc73e83f9405c82f3d5fa3170c748aa628acbb |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-f691c946 | sha256-fddd8cad30e2712f948f269f5a6a6facb72e7329066a3c593cfa291c13e0a106 |
