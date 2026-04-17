# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d68f21a3db07e083abd55cb8f30309dffa35aea63874e95510f19d0d69cb1ce`
- fixture hash: `sha256-370af296b8752ce6655fe59921b05e957209333f8adae37b056699cf10a9af35`
- score hash: `sha256-098bea7d6c7f20b3a978e50d09419af11fde9f45aaf6caa2220238032107c4f4`
- bundle hash: `sha256-0d6b0a70b0982d69794eae901bf7a251db57e56f3575b4825dd8d052c0140e77`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13ac363ce7285d3640914d39071894fd6c80687f14f6807f8531ccb47249088 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7dd1f0a44f10f6ad33cdf6adf40f92dda1466dfebf6d2a8e124a7a646defd2d9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-59b203e19989c786e3026a3530c7bf700075a5539db5bc7af8694f718070a403 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-74aaa5a1bd4373f54d5c1fcf3f0454abe21eec1b7ee5bf2b6651a8be2af32b9c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-923c552b | sha256-cc2ef599560bca7018ee5487ce6a307f521a48811da7f416c65903022be86f5f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-923c552b | sha256-6866efddc3a5ae0c4f63a0603ae46f2b71a84ee2fb32361c6d5fe542f2feebac |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a05a1cb6 | sha256-fe88532a0978c8fd61a08f6a90c198a6450080b01654333dbfc1b19c8bfafd39 |
