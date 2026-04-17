# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147`
- winner mode: `graph_prior_only`
- trace hash: `sha256-12b53203712e88b756dee356041b3ddb0e18e328e1c8f8ade691064553599eca`
- fixture hash: `sha256-8ac6a4fe3950f0ed5cfb2e1b9bd9c7ad4d79faf9e22bb913250d8fa59920cf2e`
- score hash: `sha256-3ea6569a7a61927f89c8b1f4f5dfd9e783ee181c6c140c1a69212f2f02c2a431`
- bundle hash: `sha256-f5614fd48b92d7fb912790239aa6b4a0f6580d626f5e499fd8a5190a4212161b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55766afad53c9e202670418bdf755c0f71228a26fa5f954c36b74006ec3fe092 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-49840d4bf692fe29aed64e10507afd8d99568eb09a1847c1bedc6c6a6c4cc082 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f0efff4f46cbbba814c92888ba7e03bfecd363ef85b307a5a0cefee99c209793 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6d5e738f8ebc973a409cfff17f66b39193455e295bf5ba130f380a091947587c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0c857f8d | sha256-dc090a5b7da313a79d6500480e0ddcf01c316317f750be07a14c7e86c9708804 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0c857f8d | sha256-1abedd705e4de0d4712038459547e012763495a69384f15c0fbdc1c6970f95d9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-103267ca | sha256-52f01c311e04892bd20d1c425204f2953bf3a9694734d8982bb11767edeb5f1f |
