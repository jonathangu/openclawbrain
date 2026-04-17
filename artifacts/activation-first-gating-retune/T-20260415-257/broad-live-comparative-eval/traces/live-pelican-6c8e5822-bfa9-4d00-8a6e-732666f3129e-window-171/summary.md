# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f2aad77541ac9575f5e5ca17b331150d26a5ffdab9f43024542cda1cc603e5be`
- fixture hash: `sha256-bd1f8b0e0683d35bf0b6cddabbcb17bfbeff749dd6d56a3da4fa75988fc68560`
- score hash: `sha256-893ca260abf348996c1cc1ca89618e2ade814b72af3d5a4ca67ae2c636f2b108`
- bundle hash: `sha256-f04c166a956060b3e3f5b84515996543385f354886918cbebbb0dce30f5a5952`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b6dcd51a56bbf9edfb3ea54756a6521b5761e2fe2a8b04b095719a90cd986e9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9642c4dd01b84f5bb12a3637ea22e9e96c822e03419920cb0c45d726685df4f0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f94d385c069c697e34f5223aa3c22c8071dacd79e71f3f38d83310ddf19c933 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-486e561fd7499ec40ba68c78c37f777b28b3e15dbaa14349b0a9fa266dcd3ed6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-63ad3045 | sha256-509f314233e67d6e0394abece86491e6bda6af8a06dc988110eb67a4f295594c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-63ad3045 | sha256-70d92ddaa849ca2aa6d466a4a69b6d9ac3818028184aa46990e7605e67869287 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0af67022 | sha256-99ff6617037be68596fbfb6941ead7753a9803a1a5abf52c3ae4c7b02afb5de7 |
