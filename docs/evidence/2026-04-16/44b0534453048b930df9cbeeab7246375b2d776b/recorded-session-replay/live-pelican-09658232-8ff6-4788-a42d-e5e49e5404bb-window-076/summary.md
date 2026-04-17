# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e90ba91fa2d821b34e7d50d49031d2ca2e725469eba7413ed1eefcf887d0f975`
- fixture hash: `sha256-b66c57ad146f945a1113822081ae1bceec873a0abb858cfb6bafe580d07b22c8`
- score hash: `sha256-b4563b3222a6158ee47007b799e5f799e5f830b60087f47c7a7f789fe662dd57`
- bundle hash: `sha256-fc20c8e9983d13761250f7c4efeb2a28a4056b9aaeb80a4a927f21dbf45778da`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d7a8f01e83ed8ac33586c073703951c8627b99bf4e9aa0272b865992ce2738f9 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b8df0146aa1346ef2a5a9450d19f53f4714851acd6c937fca672ed0bc2115bf |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ff02c08b3daa7a20dd64b3c48eaac54a07f4043933190d97bd208d404f917ec9 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-65189c544c36ff794fd704d78894109f8ed2b727649949da24d87036438b7abf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4901052d | sha256-272f0e60e56ad0e23b47ec4e726516d661068fca4bd8fdb7776d1f7af230f3c2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4901052d | sha256-1efbc06429bafb31de4895780b7db195561bb798397f8011848fed3eba386533 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-6d3fece2 | sha256-57db52c57212f8dfd9a78f9a727635dc5296dea64add0142f938a902dc15a1f7 |
