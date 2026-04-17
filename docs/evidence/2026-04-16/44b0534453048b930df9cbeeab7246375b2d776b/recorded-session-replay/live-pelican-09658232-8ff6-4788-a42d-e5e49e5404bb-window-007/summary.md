# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-945c2e59668de577944ec7fc5dd5f9442c630538679d6020f6fdac64e2a21a17`
- fixture hash: `sha256-19b533ee2cadb7bef94e2f868a3d98284f247e98f26920f7fea15136681e3d11`
- score hash: `sha256-168c5128651625a0e05f2e8a34bf771f17872c0ac26a5d038241da90ea25757e`
- bundle hash: `sha256-4c63872c605e3ded2bcdfa62a201fe8f8b42ca0e07b7f779c02e3fd8e349e979`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d37994d9aa92d2e1c7fd5cea54b3093f268f662f580c5608c088fa86597acbc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-45a208d82488d0ba3cdf711bfc59e66069d9c73dac5dffc7713ca0c6c8edf725 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a4d9d84367dde18c52b7562b72b22580e1cbe98d1100a83be4ce29d719b1fd12 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-1a59a9622ec69624c82c744c8c52b03f6ffe954027c7bbd3ee5fbc6766fb4f6f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a51c413f | sha256-0f7ea183237a0c7f958c5d97b8bbbaf4501124eae950cfd6e964989b24d7da2a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a51c413f | sha256-6f3abf885dd8feb598a58512654ce194900c98c8563d9de9ef4ba06e1a4717bd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-6c1c2acc | sha256-c8b7fe476005efb635f39d33e25507b175473116bd9dbe7a59c423418edcfb77 |
