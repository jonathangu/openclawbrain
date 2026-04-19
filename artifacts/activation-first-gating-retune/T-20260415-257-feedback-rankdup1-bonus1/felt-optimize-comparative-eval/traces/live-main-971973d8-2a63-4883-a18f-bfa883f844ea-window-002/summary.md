# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2684bc9ce52da3283e7e269a65aeadfa9bb4bda12e0a5937bb82b4e7e3f59ace`
- fixture hash: `sha256-4296b198ad4b2382e867baff61985bd607aae4ddc54e4c60ef5ccb597fc35e68`
- score hash: `sha256-b06e508da25d113b2bdc3ca55e2d4da8b0fc396b3c5b3b16711c9e4f5556900f`
- bundle hash: `sha256-4e5c891e03ac9d7b90d722cc1378426acc92632da666d91d720a6487a70f00a5`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d2479b1210d374fe06946cec83ff362b307da973ce6e0c46c380449deb18879 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-648348ac582291b99de74fa28085b6f58c3072df544d533d45a8bb1fb2961b2e |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-cf655be3d726532db32fa5636fd50a100806bcbd471e999c6035a5634febdc13 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-393e64b5b72292e49c947568690da5166a175b58e46bf7dec0ab28840eef7972 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3a14b77f | sha256-cb3d7154fa6a4cd66a9003f18478a602a3879a5bf60a7ab714d0872e70898118 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3a14b77f | sha256-10aee6409e153778585f4ded203ce6c9fecb470e062b576ab57b38935e9dd809 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3a14b77f | sha256-13d5ca52c1cb3d46856fd86842811fd3bca184c6522a6860d9ef7a347b4f6a92 |
