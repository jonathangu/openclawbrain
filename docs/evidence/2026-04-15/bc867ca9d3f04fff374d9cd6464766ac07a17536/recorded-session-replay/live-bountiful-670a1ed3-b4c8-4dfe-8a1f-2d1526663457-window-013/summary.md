# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad8a200767aafe991ee7054e677a19f37758804d5a9a487f59ccad4263c83187`
- fixture hash: `sha256-23ce3445f512fae9ac35202b97a34c12c8d0db3c79197541a8b90358597638a3`
- score hash: `sha256-f8693e8ba18860bf2ba3119f8fe7d90c509c86cd8838d2e0fdf4635e60a4b842`
- bundle hash: `sha256-8760a72a7919be91be2f4d90d8ca7162f0755877d5fd3b7eeed9fbb365e7326f`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b757ba8fadc84f09e0e7aed31f0b4ebd54fa8fe354fc559aafe046aa0541083 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ce259bdda4827506a116f59657a42d212639423bc371dbf87335e103b08dbadd |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5f30372355077dddfd65f85f92f15178e6ef0715f7b359e31c34bae5c7964a9c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-d580aaa1fb48ae7ad02e36b3244b9f2c255362a4ae450781d04e75575599cf1c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5491493f | sha256-9e750e58247c716c07b88572799e00771eb50723dca243e91ab258d4f9bc8a21 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5491493f | sha256-f58611d0b841c70e9a77db884f7981e5661166ecf732172666e88968dc49cf2b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-5491493f | sha256-9e750e58247c716c07b88572799e00771eb50723dca243e91ab258d4f9bc8a21 |
