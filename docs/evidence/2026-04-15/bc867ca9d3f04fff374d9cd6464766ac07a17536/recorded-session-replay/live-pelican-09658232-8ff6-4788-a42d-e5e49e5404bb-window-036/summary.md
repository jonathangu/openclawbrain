# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d855fd526b0432f6da4ae83a914585ce6467161a22fe45c628b20919e2994b08`
- fixture hash: `sha256-a059e9b8611b556f3c483b97168ab252147668d3316414532e38d0791f5cd0c4`
- score hash: `sha256-7d0a223520bd0062c76db44e5dfd0c8c51a516629575ac3cc5286790a914e4a6`
- bundle hash: `sha256-5dcf44fb33ef4d578a434e72a1da89b90628a2c6362ebf9a2ded3e3bd147eedb`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46808c7f90eba103441fec044b9224d9dea48b85cde7d0c53efec734a800db3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bc3f0d8157547ba288ea23d49ddc41b6c3323da3a56e7fa9aae584c2b2a7985d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-06d3ac1ce1a28061ba66aabbac22cb8b39f503a28450128691333c0bf0f1b304 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ea7f9173671830529960b258481d762f13aa183ecd9f71f273da33704da63b0d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-09ca6f51 | sha256-e7b8369927b8d9eba8f9cf1db332cf60a9feb3bcabb9f79420affc0f5fe47d66 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-09ca6f51 | sha256-46b98b5e457f437b2c9062f7ba11333bc6fc825bdca147e67afcd07411ac55f5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-09ca6f51 | sha256-e7b8369927b8d9eba8f9cf1db332cf60a9feb3bcabb9f79420affc0f5fe47d66 |
