# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5dd6be200875e55287b6b027aec6adb1211d2b987b83c6a37b985516f7118529`
- fixture hash: `sha256-eec602e66445ff4dd47c7240e799fd3d8564ee87f3fa97f5e6b5673abf356c14`
- score hash: `sha256-db565317f6dc06166c39f3015e7689f87c7b719a6b0503381ac50d69555e6170`
- bundle hash: `sha256-e23311b77a5831d153375a362bc531e28df49d4d9857d3df68eb8c894a89b61b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d99ddcb432bf41d8fa10f8ab6904c40f835adbab6565ab293b9f4c7f5ab02130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c8166a5320eb6030b7039eac252aa336d6509c8bb0bd59d1521cd925277b9bbe |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ed5a222d7c7f4c0f2459fa9c503e165c3f5f4b8bd9c67b946631030593c6433d |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0a1ceb83b7ac718c1a64441ed5b1e1de61bc80287a8e9b03e6670e23d78a7b67 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4def02f9 | sha256-8efa554d15fbe0ba8859a35f276dcb31cce72a19e69f0fdd41dd5ae6aaaa0f6f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4def02f9 | sha256-a70c62139d0cdf8097b07fa1756439fdc7cba126c742d0b1ac24403ac7747be6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5315741a | sha256-db85787c236f917349efb7bf0b6d96479f00e9c06f16a4175c0e9bbef6dc49d1 |
