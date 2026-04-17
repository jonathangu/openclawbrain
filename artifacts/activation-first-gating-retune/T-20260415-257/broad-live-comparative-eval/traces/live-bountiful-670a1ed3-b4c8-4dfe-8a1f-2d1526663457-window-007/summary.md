# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-829093e68c8b369222680a9ce88928380b9f7729d760e59ece8cf8d1e776b82a`
- fixture hash: `sha256-31ba926396eebecc30aa75781e7d614cd75f3d45744f5fc68d2426d0829db138`
- score hash: `sha256-42ba262a2e0a0e7c343ad84ec20b121e574f7b013055a46291d43f7e68bf6990`
- bundle hash: `sha256-1f779fd11fc33810da631e6eca39e416f2a3a46cd144ac20949205b2c3c236f5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8d38776b0b580dc292e4970ed98776136ff0d2acc01ecbb7a8d527a0c51a84c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0c2b1fbb7865742e9d941875abf962508bf902d99408f651bf4bdd108ed717d4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-856f2f75170691663e109c6d2be857370a5dc95460c2bed2456671fde63cba70 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6bb1a780b220974785393779275014dafc053109468b8527d1c45473a9a2687e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e2c9d1f8 | sha256-81bf2f59f095c5596174ac8171b4586b675a8ae9f25e5f4da737f20bebd8c796 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e2c9d1f8 | sha256-07f5e1b794205068040a74b3d120dd17eb92accde2ec4a926297139661936bac |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-84f8edeb | sha256-9c57a7bbd94e1c73801f91fb03e4ccca01960fd6ce4fb4871eae40c510e30cbc |
