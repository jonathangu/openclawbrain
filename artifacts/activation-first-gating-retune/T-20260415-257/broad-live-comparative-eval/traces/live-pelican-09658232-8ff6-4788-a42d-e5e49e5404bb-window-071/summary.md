# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-aa6afe07711fbc8a13484cd14e70ac82c78cc503ee5449452a36b775fa63c3d1`
- fixture hash: `sha256-bacff39860081979b6852dc7223e7e30d3e6e8700496899a8864e78cf3c36fa0`
- score hash: `sha256-1a86f489e638639bbba1d274feb464badc4be6f175c1fd2ce9dbab01135e4b26`
- bundle hash: `sha256-051ef0ae69196c0ccccaaf4a3c8e53686329540bfe61fd9029dce86690315f8f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-080c0791d3c8d4b27935c18a06ca48413df84ee848ffe0bfd6099d007a81a298 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1f71a2d78947c04726a5b87200c4bda5000d222c2a3330d994064252cb9c7998 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ba5500f9de897ef62253bf79628fcb53bd36ee6b2ce67e7de63e6362b183a7ec |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-214a8f91b7a0a932efed9c8716dba954f1612841671891c6ccbe18600f8f72d7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ada4520c | sha256-56fc9c4bd70a5a690ecb87e7663409d02fda808cd04e1823d60e6d129c778e7f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ada4520c | sha256-777b001ddb4c5a6a5eb15025fc3ba991a1e77f5a0ce0a5216a1cceed5dc9d3b1 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-dd31e829 | sha256-5141ececa15cf38346cb70194c9e7d818e4fc161f466bfea43aee1fb50431eac |
