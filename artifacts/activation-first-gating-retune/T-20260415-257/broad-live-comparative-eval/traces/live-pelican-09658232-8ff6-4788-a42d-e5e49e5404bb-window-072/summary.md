# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ea52ae43f846ec5905bcf93ba8cbe911779b358315c7c925dfcb3bb8d88c42d`
- fixture hash: `sha256-5ca126c8a28da22685d19c74b8dc7e5cb0bac37c0b916d2162c68f83275f6394`
- score hash: `sha256-44488f95a8325f07907e970c84a6589972352f961c5fde17aee8b14965f45ad6`
- bundle hash: `sha256-175b5970503d617703f29a4f87ec306333fd2abf5966d469e8b32b025826281d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ee8e31cf8824d425cda16aa09e9361fc2028a17b7c4fcbcc21c2fa64f147edf |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-32dfe11511714afb83c6c13e4e5b43750d0be5a88cdfb0ef0f3c949568b609e7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d9f01e93107ccc7318760799db2eb5e934bdb0f8fa786bfc2cee1a611e211e11 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-113ddec467de8307aab8863949239e240bd5d91179cbbd5d06716b83dd0f1ddd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f20c9bdd | sha256-bd747933592f84ccc27369163e826d65f25703f09adfa655d237dbbb1c5cb698 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f20c9bdd | sha256-6b4060c545bb9367c6badb55c27ddd2fc63a7e5ad8f9e13e9738629fcb96a8a4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-21dccfb2 | sha256-cc00aafcb1b1e2c8c701c151425060e66a610c06931f2198d2ff71f59597a4c8 |
