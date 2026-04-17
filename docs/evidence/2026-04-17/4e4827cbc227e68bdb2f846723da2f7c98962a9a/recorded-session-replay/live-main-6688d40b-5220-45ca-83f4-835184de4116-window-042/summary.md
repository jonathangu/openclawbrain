# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7d50a8bfbe12d6ec52d00a65d5c5309711fc92d4bd65677275533c95c1fbb9f9`
- fixture hash: `sha256-486866769a6220eac0c25d8477d823ddd1d78a29159bb789869bb12cfb7c0a16`
- score hash: `sha256-d62bdd7e3a1a7339cba3fd4290b0addc4f956e43a5001f59439082556523bb33`
- bundle hash: `sha256-1a485d05e295b824db377d45a5a598abc68e7c2491b04bc86ebd05aacfca4013`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-030b3c11ef3b6ff56c24da96c3a7b6b56306fdfbd30d56345e3f6aeb18dc6984 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aab0118f80444dfb7e72eb084f5446bd3482d2f5eb66decf949e26ecb671c772 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87765a11603dbaa285694ed2a6d523b6720f63131aa24bb3c7bc0ed5bb8e481e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c149a7a2d46e9649c37e3f72aa3af93a8935cfb30a8212f73b25fb632de0cc1f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fb8d9fb1 | sha256-1ca9620b722233e06e007254e6c5c7e4ece576f54001dceef041ec32bc35b5ee |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fb8d9fb1 | sha256-880d6458497aa8a5a83c642bedee2d3dc88540e53917a56bf37b090aa257e298 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-63fa1002 | sha256-b914bd932c0fff886e25500273c3d2985598d63399c630c1bf43f6daa589dffe |
