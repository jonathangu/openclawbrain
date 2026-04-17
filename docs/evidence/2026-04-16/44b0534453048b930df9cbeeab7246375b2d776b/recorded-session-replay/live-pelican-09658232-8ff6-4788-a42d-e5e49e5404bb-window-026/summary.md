# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6b834033c55b0c1fa4b1b699a1d8df0de8cfc6135b0ef6c168d6806431da077`
- fixture hash: `sha256-da3f651615d0891ce7f18953e9f938d1c01935c73ef17cad6fec24cf102a80c5`
- score hash: `sha256-cfd46c757de66d485cde6540f0e146b1d176d504bb0ba5bdc689e92db9d30e1d`
- bundle hash: `sha256-2250d05681c9c7d0cc4f6bf579f77655ab8a51b9e4bab866d822a80f7fa0bc53`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7bb32e7c3cacde4e347cf09cbdabeddff358a8d0a73f0b6ff3c688033dc4ad3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8f7deee3752ba5bcfc3b449dbec24575d7b91a8d6849cb86d97b3b7ab461b5c1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38ef4803ee4be73264b444b1f74db1d8e0b6d6bf387ac922b0dbd66e4fb0cbf7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a65ef8725111d5fc286855941cddc755a95a61f325a0507bb91bbd04be74538a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-52ba78ff | sha256-797fd7c30b874f19903f8c4aa95be4749a27d4a7b5a8f0e5d6803a7278df3352 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-52ba78ff | sha256-a21e3d316da8efaeb100a163bd465bc89a312bd5e9f2ab688f5d07f04faa663b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-37606fa2 | sha256-c36b8a530f3dced8c337010e1301ca8a0898cfd8a3fc4e63ec671285e1bff61f |
