# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3efd8d20cb1c9f5888880c948560443e245c94c38e7a8aed8360e4337502a229`
- fixture hash: `sha256-30c95d9f64f0e6f7685627b63849a05509d84fe08cfeea8b94bf36afddd8cab4`
- score hash: `sha256-6f358d8cb7f47039f4cf1f3589abcea663de9a1417ecdfccb900ee9febb2e200`
- bundle hash: `sha256-86f99db12301f82b7109fd376bffa68e9cc9544e7c6981e1ff71b0e610e59725`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af57ec3a983f1e66674a8f934086aafaefea804fe02b97ff477ffa64d3b5023d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dc31afd3317b3282dca1a11735fd22c51eb6b7314d1fd6a1134683f4c16cdb88 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-97357f55e1e598808309c222a94f402f7cf16f836d8981c8ecdd64a94bd0a01d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-0d3677dda61c6b041f77378f58c721e9794b12c1241387cafd520ae57b21dca6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ca21e2d7 | sha256-ddbf081c1921f382f1f7dce8ba87269642e6e91ddf93ecb7b5f208b5c703b27e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ca21e2d7 | sha256-94a12415ed8b931acde38108581038707b21722a9da9f6a48213a3adfd57d9b3 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1ef29932 | sha256-9d1379c217b98b5dc3593e4b59f577372e40f3b11fb22d99d57e8350854660c9 |
