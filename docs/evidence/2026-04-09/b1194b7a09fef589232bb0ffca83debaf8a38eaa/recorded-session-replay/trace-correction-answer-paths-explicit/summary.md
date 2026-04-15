# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-answer-paths-explicit`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7675563cf859465dbd888bb06d4b86fedbd83e541945091d1e8df5ee12e84c1d`
- fixture hash: `sha256-57a7a6e1a6991f696a856f4fea90684928d5f3ee0f026ea6b5951d4fc10cf426`
- score hash: `sha256-9bfe2a23f653e51394d996a48de7422e433e33eacead19db6384b3403b8d21df`
- bundle hash: `sha256-c6b5d0a8e8d1079739fcd64d9c82a4bd9b1a533d6a22bea47a3ed7a700b06906`

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
- phrase hits: 9/12
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
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-2042863833d6ce7ab296d4ac789fc38e8c68b0974c203443b4c7c040ba9c0cb6 |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-3c7cd0479949e8157100ccfd51c389ec05a6af3f195737c300aae1d64d56e3d2 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-d8d9de3bed36180b3e7fe83c21a574d9e7c5c78216d0b05894ca8f76dfad4df2 |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-5d2455dec45b9f078af37ebf43c84877011799aa93ec398f74f67063cfe40920 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | explicit-paths-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | explicit-paths-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | explicit-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-8b7a5bdb | sha256-fbc68c6466f5fe03f312a105f32adf1050b4c72bd119d93b5324bdd771e343c7 |
| vector_only | explicit-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-8b7a5bdb | sha256-fbc68c6466f5fe03f312a105f32adf1050b4c72bd119d93b5324bdd771e343c7 |
| graph_prior_only | explicit-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-8b7a5bdb | sha256-fbc68c6466f5fe03f312a105f32adf1050b4c72bd119d93b5324bdd771e343c7 |
| graph_prior_only | explicit-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-8b7a5bdb | sha256-fbc68c6466f5fe03f312a105f32adf1050b4c72bd119d93b5324bdd771e343c7 |
| learned_route | explicit-paths-turn-1 | 100 | yes | 1/1 | no | yes | pack-8b7a5bdb | sha256-fbc68c6466f5fe03f312a105f32adf1050b4c72bd119d93b5324bdd771e343c7 |
| learned_route | explicit-paths-turn-2 | 100 | yes | 2/2 | yes | no | pack-4a1a7f90 | sha256-ec78ef07f14ef5a198cb756a2d12c5f5e265b2f08d46810550207a24fdcacd54 |
