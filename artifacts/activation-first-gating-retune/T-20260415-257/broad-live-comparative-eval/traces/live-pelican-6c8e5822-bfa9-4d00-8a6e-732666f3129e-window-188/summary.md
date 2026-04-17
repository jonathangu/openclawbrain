# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2286f1962f858995a9d11d68ccd4ff744be8c0925ede8b9595870bdf0f8216d1`
- fixture hash: `sha256-7131fce3dd7f89b87927812976c9719dadea253e34115e7f37e0887827e9427e`
- score hash: `sha256-6cc236caaf5216d8ad9fc37f9f074bdae55467b0791f3928324d10dc09ff8cea`
- bundle hash: `sha256-219fe10ac67835ce1fa01e7c673b4a781384dd8af31a61f85c210652af032733`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cce362e58248df18caee133abbb86ea37c7c8cc312d9027b572d5a719da7a87 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-44b94cf126f917393b205eed98b4fa8fb2c731c38a1cd9f5434e7088365b03e3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e0476594158bd7273a2560928cad56aab116d1191f853122bbb7275bad657d0e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-faff621e03b1779ec39054aabd3fd7778902ef169e67abbb2c18eb7b39fc6bcf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b67f07c5 | sha256-e6f642589ea6fd3babe111f0007118e9d709e58c80a221a20313fc3eb0b6c217 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b67f07c5 | sha256-f8d32d3b763a21a5c8b7d034a11ec6a645fb535ae1eab21af4eb268ad1c106a9 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-89db73c0 | sha256-0c6474791ec56d0a213c9f9b585c292015d93319fd70db95f153c96f717294ca |
