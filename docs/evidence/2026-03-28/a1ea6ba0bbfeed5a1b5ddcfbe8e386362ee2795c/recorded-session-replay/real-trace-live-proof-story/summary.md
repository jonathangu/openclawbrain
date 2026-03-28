# Recorded Session Replay Proof Bundle

- trace id: `real-trace-live-proof-story`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4c1568a9ba0750cfd390d01b493f4917b1623caae678690ae1ba9d56044b2f49`
- fixture hash: `sha256-4afd120b7c502d3a7d8b273c1f2f0625b9a6b883c26f170ff038034ace7e1ed4`
- score hash: `sha256-7c16863648bdb61f22b1e5d2d40cae2619c2a95aebfe67f318730bfef872d08a`
- bundle hash: `sha256-9b30f4ff42aa81e8e85bdc147f148f3e9610e9984cb489e69534bbb56f83cd8b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 7/16
- phrase hit rate: 0.4375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 0.75 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-b55d6f80872b4358f8a0d4c3258dc29907a479488e5ef550fe7039d3c93240b9 |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-a944ae2b7780112f60f5f008db9aca2596e6600776953d77eb9d10954454f276 |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-7673643fff3307b0df9de21be7c164f34d1514f90a1919a2509ac7f2b2a2cc42 |
| learned_route | 3 | 3 | 3/4 | 2 | 2 | 3 | 2 | 0 | sha256-c85452fa9d4ec079f740862ec35b9fe891e100d973ce4b2971257bab72923457 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | story-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | story-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | story-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | story-turn-1 | 100 | yes | 1/1 | no | no | pack-7ca8f7db | sha256-6904e261061b7b07130678d137822eecda89b7edcc10696f373b7a384138bc28 |
| vector_only | story-turn-2 | 100 | yes | 1/1 | no | no | pack-7ca8f7db | sha256-68e9f45a85e437ace33e9b881de6cc3bd938b5a130127d739646638b65edd876 |
| vector_only | story-turn-3 | 40 | yes | 0/2 | no | no | pack-7ca8f7db | sha256-6904e261061b7b07130678d137822eecda89b7edcc10696f373b7a384138bc28 |
| graph_prior_only | story-turn-1 | 100 | yes | 1/1 | no | no | pack-7ca8f7db | sha256-6904e261061b7b07130678d137822eecda89b7edcc10696f373b7a384138bc28 |
| graph_prior_only | story-turn-2 | 100 | yes | 1/1 | no | no | pack-7ca8f7db | sha256-68e9f45a85e437ace33e9b881de6cc3bd938b5a130127d739646638b65edd876 |
| graph_prior_only | story-turn-3 | 40 | yes | 0/2 | no | no | pack-7ca8f7db | sha256-6904e261061b7b07130678d137822eecda89b7edcc10696f373b7a384138bc28 |
| learned_route | story-turn-1 | 100 | yes | 1/1 | no | yes | pack-7ca8f7db | sha256-6904e261061b7b07130678d137822eecda89b7edcc10696f373b7a384138bc28 |
| learned_route | story-turn-2 | 40 | yes | 0/1 | yes | yes | pack-3b71ce3a | sha256-aafb019a09d3630ca18208df78ec7493de6a680cd5760397664f8c068337da93 |
| learned_route | story-turn-3 | 100 | yes | 2/2 | yes | no | pack-f6f18734 | sha256-a449a88a75117f858f3ef06b34040dfc7470d8ff57edb5fcd7bc4f9ede843d61 |
