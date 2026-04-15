# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e90ba91fa2d821b34e7d50d49031d2ca2e725469eba7413ed1eefcf887d0f975`
- fixture hash: `sha256-b66c57ad146f945a1113822081ae1bceec873a0abb858cfb6bafe580d07b22c8`
- score hash: `sha256-740993a7cb1302fda1e282052f7dd312e687813e352cd402b39ecb202f892aba`
- bundle hash: `sha256-b625229336f1bf859ddd041f4324daf2451134a69528cdb9168916af7028df01`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d7a8f01e83ed8ac33586c073703951c8627b99bf4e9aa0272b865992ce2738f9 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d2b6f1ff5525ff3196091fd0468e33e35c102d2ccfe199638998997b095ae75d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-52af804ba232911fa54fe2867b5b089f3370d0310228f33b45d9eb44dc76a3e9 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-05f7c352b6317e2d79f3aa4b7f885ac974db8984237be52577a5de6d3542dc1b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-09264de0 | sha256-51d8e2812aa804cddc5968bf74c4082591576dc335394320f59424fcc3467914 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-09264de0 | sha256-f18a73a5f60f548be3b9e9ea2b4436a566814f37c5cc50cf7003ea3c35837f0c |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-09264de0 | sha256-51d8e2812aa804cddc5968bf74c4082591576dc335394320f59424fcc3467914 |
