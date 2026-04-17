# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d734aa4f8ff91d619ea2bd69d87aefdd3f36d0cec38d3997b6f1c5ab56a102cd`
- fixture hash: `sha256-b5a8d59003130cacb6d12d20cb7f35591a0ecaec31de33844db54aba06f55180`
- score hash: `sha256-6beb382cd53bca3c933696ee9678f48340b06954f04f3e3cebdbe4cff1382a4d`
- bundle hash: `sha256-dee9c264e1cac120d9fe694a67ecb204bfb804f6f2619864d07d25712167e3e0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2c822d5a812142cdf3ff00336272092554327fa9d0fe665c2253ac281723c371 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-efeb1e5d8d147b8cd021bf4c140d0c9e116560db3c5a6e362321f02ef20124f7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-99dbf47ed3a99f644d9be1d31311b003b03eb7a0f25fc27b5e5053cc3d8879e4 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b8be728245a121cba4de4e01cacad9e73979cede6e5e84f816438237df6f367a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-960b1178 | sha256-abeceb41573a50ad499c3e5449897828cb55fde26fe61a3dec5e14dea601b4f0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-960b1178 | sha256-d1a5d3c79c993f926aed6bba5440ad7dbf5f7395ecba8774e4ab0a80cc1a6512 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9cdcc53d | sha256-9ff7adf108d2048a00cfdf49e89a48d5f779cacc62fc58ebd429340e6c4ae40a |
