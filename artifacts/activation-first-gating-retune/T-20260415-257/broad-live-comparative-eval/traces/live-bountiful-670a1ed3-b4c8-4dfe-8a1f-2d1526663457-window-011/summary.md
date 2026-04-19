# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2d785c91c6c2597c88bfdefe91898000c30733ecf3cca8e1fa5fd2d6621049e4`
- fixture hash: `sha256-63b7942b83cea800c5fc9cb957ce0307322538d9d8e1a745ea7ab80b74e65911`
- score hash: `sha256-98cdf0ff93e22921a42d7229e73f116c3fd1aabedd12ea38d60227ca4ed27279`
- bundle hash: `sha256-394e91df1b56f7ed3a86f0d4fb6fb2fce089f63cc8a42225e5e2caedea63428a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4648c104e20ab98d8928f41590949536cf65a6240f7fac95811ce6126bd169f5 |
| vector_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-9335dfe450af5eb59c260617ae38049e2b3ce44a14fc3f23b1cdda096116a4cc |
| graph_prior_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-da84568f3021db432348ce612b1e24340a0a3ea2183dfa282a99021a0a019132 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-0c0f9257342fce0928d65e4103fa9341f866aa5ceb5d29ed59620aa7b34f24f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-cf83dddd | sha256-68364eec548c1877f936fac929dab6b56f9c905c3d796c19687a4e6e7f8d7232 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-cf83dddd | sha256-9615dcf9460f7f42cc8225d7b27d88b5e6ee0b002feea64a2d947596417aaaa6 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-cf83dddd | sha256-bd8bcf55cfda995497f7f542b03e563230d860400235787b2d1939edd6ded5d9 |
