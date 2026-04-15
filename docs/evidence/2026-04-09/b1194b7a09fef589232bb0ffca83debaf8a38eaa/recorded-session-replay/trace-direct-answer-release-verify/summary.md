# Recorded Session Replay Proof Bundle

- trace id: `trace-direct-answer-release-verify`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01b84ada8d6846964137ff080684902a4e4cc0e43a8cefb78c04b2a1e32acc17`
- fixture hash: `sha256-aa503d9d77de4a0a7cad9548aae31476d1a1a5a96b73a44dedefa1b9a484712b`
- score hash: `sha256-f303d322f4af33f027a2e8f7d6197a074c363857ad03e83bb47733725ca07454`
- bundle hash: `sha256-64adabc4e0f66bcabb775441dc9e27c42d630c432de383d701156acb321d3c00`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/16
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/4 | 0 | 0 | 2 | 1 | 0 | sha256-50e4253f1347265eaa9ff02792d8234c7d8ec16a6a093d8a6a69ca92ec831ad5 |
| vector_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-d78eb3bc9cd0a9802ae909154e502789fc41d7cd8d256f68fe520d53f0cd1120 |
| graph_prior_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-c4ea09c75c7f1b7b0f3df872051012ceb1c7a4f6cebe52234cef46e6489449bb |
| learned_route | 2 | 2 | 4/4 | 1 | 1 | 2 | 1 | 0 | sha256-f92dd2e7b700ebca3a940d1cf0872a3f2360e24c5e47ab9fae53814e043d7732 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | release-verify-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | release-verify-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | release-verify-turn-1 | 100 | yes | 2/2 | no | no | pack-c85fd2f2 | sha256-ded840d5e33b84da9e60b34be3fdce1bd4bd03d71602b9483ac709767539a0eb |
| vector_only | release-verify-turn-2 | 100 | yes | 2/2 | no | no | pack-c85fd2f2 | sha256-086b69db9fac010101fe9fbcffa3f79cb2ccde6b0a363b0bd78d16f9b1321d2f |
| graph_prior_only | release-verify-turn-1 | 100 | yes | 2/2 | no | no | pack-c85fd2f2 | sha256-ded840d5e33b84da9e60b34be3fdce1bd4bd03d71602b9483ac709767539a0eb |
| graph_prior_only | release-verify-turn-2 | 100 | yes | 2/2 | no | no | pack-c85fd2f2 | sha256-086b69db9fac010101fe9fbcffa3f79cb2ccde6b0a363b0bd78d16f9b1321d2f |
| learned_route | release-verify-turn-1 | 100 | yes | 2/2 | no | yes | pack-c85fd2f2 | sha256-ded840d5e33b84da9e60b34be3fdce1bd4bd03d71602b9483ac709767539a0eb |
| learned_route | release-verify-turn-2 | 100 | yes | 2/2 | yes | no | pack-f11a497e | sha256-4a61f50d06dc3ec7df04b6563f875ea6bf065d2d549c5f3d82e1485e1b9a9d46 |
