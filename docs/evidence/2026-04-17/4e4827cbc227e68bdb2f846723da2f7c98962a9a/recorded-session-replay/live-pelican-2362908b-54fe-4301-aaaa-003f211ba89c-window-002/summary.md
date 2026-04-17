# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df541968ca52654e5efa48a1a6713bb4511f8366d389ef30e36174b0478a0f72`
- fixture hash: `sha256-10a1d9d424d59bf74d6edef2d25c3d9864b38e04e75b6ff4b28dfed92245cd1e`
- score hash: `sha256-8bab86248696d72e558d1946bedd6fb5e01fc1b9e55cdca55cd0f2b56bcfc3f1`
- bundle hash: `sha256-7833563810b1ab3b9ab60a22652457efbca83bd6eef9b07cfbe9e63921498f6d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe456f8f9f99c14a6c26ae3cfe1240fa644752e272eded6c0df3fca37912d301 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9a4b136dfc6921dcfb56232c40243cafb8ca95c1f1e5f799a0e6de8de5fa8b45 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fddeec2a4465ff90238ecb1acebb51648055ca295ff476deeec348e30d738282 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-be494716f2785cdccb22131105779c5c9d5aa2ca6e3cc089185b1ac170f32fa5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dcfe5f4a | sha256-dfbe9a4c28dd4030bcb4c74393fbd14841ef85a824af3eb9e85021786ad1fbb2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dcfe5f4a | sha256-31039c56fe951a726f333f635a939108cf1f2b511b149498fd461ee45ff8e023 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-847d4203 | sha256-f12b217b5eb553f6d3be107ef7d3a9c625473e96bdfb4f6362c8788878b99b0c |
