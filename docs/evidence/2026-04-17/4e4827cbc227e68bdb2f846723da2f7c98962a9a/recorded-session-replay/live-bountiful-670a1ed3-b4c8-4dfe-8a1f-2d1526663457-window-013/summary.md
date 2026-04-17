# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad8a200767aafe991ee7054e677a19f37758804d5a9a487f59ccad4263c83187`
- fixture hash: `sha256-23ce3445f512fae9ac35202b97a34c12c8d0db3c79197541a8b90358597638a3`
- score hash: `sha256-ed2eb58dda3c2eb85a7eccf3c44a521eac5a39f622675f9b05b406d5ba0afb09`
- bundle hash: `sha256-1970840abb62caa7b16ffe8a94143ac31c0a21fc4b03184dbe5f82603b74c4cd`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b757ba8fadc84f09e0e7aed31f0b4ebd54fa8fe354fc559aafe046aa0541083 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ffdfb19a1be1c59d4a2d9f3fa07eed1ed8b87ba4d09334ec1663f48ed415a199 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f7a600750a73e4e7626443074446b407ec7043df76161e98f79117b1559cbe57 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-dc8765952318481cb60f1470ea7881f71c082893ea746937dc7be8ec463b8652 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-870266be | sha256-b31a48cec6de5a3ce4550874b09a0a4d74aefc93dbb2e13eadec6d1069ad9d7c |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-870266be | sha256-8b15c4f8fecfac349e455cc57c38fdab4ea874f3193b84ddd3ffbc710f7734d5 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-11c75d5b | sha256-690531e48559a7295a1f0bd424c15a95cb2325d0aaeb2da54a4e7bef848d0e10 |
