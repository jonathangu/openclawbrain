# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56cbb762872ec9be931577e0e0cff6303eeba2f34a75a56a7b3caac6fcb77a1f`
- fixture hash: `sha256-94342c3c881533866c5dca496c9a26c188cb0d64c2968ed52c2a79ce1e516ec2`
- score hash: `sha256-ee13003e80d1e4cbed05a0bd3c93542f8586f633670386535711455086139e64`
- bundle hash: `sha256-43ccb5aad49974da3a27c7edf8bb8a85c7308e791234a0e64ee7be1e5f1780cd`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b47b12d5a1c7efa8df449fc327f742bc9ea2e78e636eae25d8d3c474db1900a |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-e2944b2240ffacd2084f194e40415b99a1debb940ff8a8c9755ba7c71840718f |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1820f5ede55608343d98230a8859bd1a03861de288b3252ade8345f8f65afb24 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-af8faaec34ccb88b5c94d5ba0868548cfa8cd0132349c16da8a3a2cb316600f6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc17c331 | sha256-6c1c52ec0e9c8431e46caf6f81b4374c3415af043ea4d70b96c8dec3c19ccdf0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc17c331 | sha256-50f2fe341ed2c7ee16fa307e77a704bd75b576ec66229dc901609173820ac563 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc17c331 | sha256-e36c712da2c3ce69b02c8c8fd9b5f2793cba07a4c6b59d57d90d956466e223a7 |
