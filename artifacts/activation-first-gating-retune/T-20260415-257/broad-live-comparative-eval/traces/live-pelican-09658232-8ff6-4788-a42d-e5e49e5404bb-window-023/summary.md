# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f60100eb1742bfc6c299af2f5afe9b6c211473ff986af1ecb211b198ec2ac6e1`
- fixture hash: `sha256-7060077aa89ea2d2ed121c14a4166c1764801c149a1d2df1467761d22c2169ae`
- score hash: `sha256-9ee15c3da2feb3da76dae783a97609b5df74ba6e9028e4cec09fc081a070fe29`
- bundle hash: `sha256-09036b84eba314b84f38cd43ec0b61f711e333f5f1a665ce62eb8631c0cce796`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1af67f3f0f2a5d2c63ece4b570453604d2bc85441d7219830f849b19b9d0d604 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cae7f49e5e966f991ac059ca1c27ffa46e7ba3a4139430fc598a6f9d9f9809c4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-39107fd0193638749808765bbd37b81a1af7eaf1c0cd99878cc11bc67e66e2e8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b38cc6da26d8ae724e1694a3e37992fed067e5d795147c0ae7619fc8aadda643 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-326d79e1 | sha256-af8f2f8a637fb1a9613fa7d556e562a9c791e02a9067599bca19d28aaec307bb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-326d79e1 | sha256-0673b7001432b58039aebcf526c3a1558af209d28498ae5d6baac7800ecf1a86 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b54caa88 | sha256-bb9c52542e2f390cd1b91bf9f6700c14a01cfae3d4434fa570262d89131c3369 |
