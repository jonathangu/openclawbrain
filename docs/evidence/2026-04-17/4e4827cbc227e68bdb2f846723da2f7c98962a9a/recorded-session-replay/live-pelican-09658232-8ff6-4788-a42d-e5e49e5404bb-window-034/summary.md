# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1cc11a021b01ca8aba99240d6af1922e93eb57eb02a859f34b6e95e9d51abb2`
- fixture hash: `sha256-25597cd73d9bd0f0d086440e05e70594b904a1395b870c35b683a0a720d202a1`
- score hash: `sha256-d76126f369f0b393f6755436b0e9c6dd627d325d5b1281f4201564365ccbd289`
- bundle hash: `sha256-53cb40d8030f56d6d2c61e4afddb3dbcf33466673e25ff0e04174dd3ba32aa5e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e0e779f7703426d9dc6e56462e7c187c6dea02bfd7bb266c409658015b58695 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fefd28e851efdb117b4bd442f3cba8a01d27f2a104d497e2a910bb2a2638cebc |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-58e9ad0888e7e13553efefd42a73545c81743860209fe62cdaa78f6461524b45 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-523089965cdea62991a93c39f1098587c82621b52483c46e19697fe5f8ab6d86 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-34cf30a4 | sha256-3fb7b25e910c7d655f1a0b5fadd58ffcf6e821df5a305a14b47bbc320e73fa1a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-34cf30a4 | sha256-1b5204cacab1583d055ab414c39222b3f9781bedc88baf31dfb135834b8ae4fd |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ffd997a3 | sha256-eb0133e504dc2d0866f9501d691fe36298c5e95b2a137a3fca25989f81220f38 |
