# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e0ad9ec13f7d5b82b36a685375b9d3d24391406d595ab3f8c2b0e0a5247f79c9`
- fixture hash: `sha256-0d1840771e0444519c0d4b5e3c3b57cee2fa58fe3cd78cd2a661af1ba4273a98`
- score hash: `sha256-7baf2ae770694c17e994bd386f16e713c7fa921d4c6cf9e829fee18599b41f6d`
- bundle hash: `sha256-3fcc06b0326dca20c6708daaaadca4f27946560b9d0c583dc25f5ee9d0d2f955`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-24e5432aabba4b367a9ab9972174d2db006f79b43849cb63eacaea39404c4061 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-18323d1fd26baa719c3d37bbaa1fc291cff0d32386f28c57985f44bec4a54288 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0ce8d0a8588e3f900d891aa75ece154c0ddf36486e9125b0f5ba483b4c2e0d5e |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-7320f03eb176496a2301709d5b0036f0e722964bfe12c54ba34ea00aac1c9820 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-54b65f0d | sha256-8b91a291a6b57abb35776e0c64abcec5fa670ae2f9a5303e3cea5c38c4edaa4f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-54b65f0d | sha256-1231c53ae0d01a43f9a9838ef00c4fed0704e53920441d60500e0cc07848be94 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-54b65f0d | sha256-8b91a291a6b57abb35776e0c64abcec5fa670ae2f9a5303e3cea5c38c4edaa4f |
