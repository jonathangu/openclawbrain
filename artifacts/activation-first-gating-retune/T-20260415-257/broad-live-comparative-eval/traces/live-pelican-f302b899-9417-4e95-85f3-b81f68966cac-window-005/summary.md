# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6aeb45f46257078a73e31a3ca01fc811e5a3a9b2828328d1595fb41ae1cb1b87`
- fixture hash: `sha256-b90901422fe4620c22145acdd76fedd90d08a07ca2636957ff33166af8db8c6b`
- score hash: `sha256-707eccece7db0ab8f93c872504b65e1c25c1c5546f1304d10d9778daece5a46a`
- bundle hash: `sha256-b69e7fa9af3cbf754fc034d6338aa59b37fdfaeeaade3e4f8a940d95b6482504`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f9c00df70ff9e588e665c6961063a6f0105a883c9e9bd2b1d2f815eef1057f7d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f0ef0c57aa4c1916347b33c4aa29455b35179cfabf89afacd2c8090eb2bbb3e1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1bf990f0042a991e8f6aacd37160eaeefe71ae55d9b8c1fbe28e8d94e98b6183 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-56587bdc3a3bf337094cac3fee850539ee1e26265f1a01b5a2c766f49c1248db |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-22e97f49 | sha256-7444f5dd6c24bc0615e80647e73f1e84e6b81042f6ab4c5c4ebc710eb0dbce87 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-22e97f49 | sha256-f10b8b68a1b654e35eff2fe1b649b6aa8857a637ef423aa1fb22d718fab6150d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-78d2aa96 | sha256-9df0b1042a8c96c4a88d45e2fd57e850f80d1314eb42b9b4711c7f89394307a9 |
