# Recorded Session Replay Proof Bundle

- trace id: `live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-93228e668a08c975492dc6af4e3bb4c71052274e3e003bc535d1e798cb5b7551`
- fixture hash: `sha256-de7894b208900137452009cdc652956a77d2f2658869966be1c1f8a47a12873b`
- score hash: `sha256-8f52fa9c38b14d9e3a62f3438fce0cce4e2f3b8d27fba089dd79808c2b982aaf`
- bundle hash: `sha256-3f7242890053d64d1e9f78767463855426bb949612219db0cca26dbbc30a21c6`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c56ee3ae997453b8eb93280de0f46e35ef0156aa279e2ba51ceb2f8a8bfd749a |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-9b7c6969f8fa56c4d51368a9956868f47dc21fed874c242a0c61e320a5aecc04 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-2bc934d52a40c4b89f56d20a5d6b312ab90c83c8d8cdea13e1ef76abb885c6a9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d24bb1a22e615783cf2b556cfa04835569743d0eb43838fb531406c8b6cd4470 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4c8a0afa | sha256-3198194c38e3afea608812b9b608b779676a53312321026198f0e29a5ff9d31c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4c8a0afa | sha256-a46a2caaa7ace2c6ce5c2ebb78a901f4aa4f386bbb40fb64ae87bbd27e2bfbce |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4c8a0afa | sha256-86f11254d7baaa2a0435b3d282384d973ef54fa6b34efb1cda91613895c63b05 |
