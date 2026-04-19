# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bf37fe09634fbaed69f393758385015698c30e4ffe16d85a6ab728cb7cfe25b6`
- fixture hash: `sha256-dc83b67fd93a911909b6e6a0822040e20903fda7a3d9b344617db1a16b36190b`
- score hash: `sha256-c7be25bbf93062066c0fef2869f571f2c58b699d9f180dcf9191b45137a5cc4f`
- bundle hash: `sha256-7fcf89521aec8da5b1bf49ddc17296430cfc26d5cd84c66931773acf1732535e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6a3c5859c5aa675b38ed66866de5ac4f6b502c35d08a72874cf67deb2a63be26 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-019240da118be8f2e5da0f71c64528ab8a69c5036318a95317f76c815da2cbaf |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7ed7fb1c0818067637136e23419f4bfed90d11f34ca63415feeccfe8cfeb5f71 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-546e10196343e62524d8f1a59257be4803e7dfba4621f66b114b9bf322f3a69e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b39f58 | sha256-612280df4d8b2f2eb0f2edd7eb2b741c2a01ba1a90fa787c420aebd96b2c2ec2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b39f58 | sha256-3fa62c9c69697fdfa8734c36942426d10608a2b1936eb9b4038fd06dd249bf0a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b39f58 | sha256-612280df4d8b2f2eb0f2edd7eb2b741c2a01ba1a90fa787c420aebd96b2c2ec2 |
