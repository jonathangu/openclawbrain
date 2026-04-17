# Recorded Session Replay Proof Bundle

- trace id: `live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-93228e668a08c975492dc6af4e3bb4c71052274e3e003bc535d1e798cb5b7551`
- fixture hash: `sha256-de7894b208900137452009cdc652956a77d2f2658869966be1c1f8a47a12873b`
- score hash: `sha256-323f4d77897df6f737066c1be486fad8f2002f1aa6bee0cabd47fc9a8cebe805`
- bundle hash: `sha256-cf5a6f709e8f8a471d4530146a53aac68e3d1d13a21970fd90672f3b9a374140`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c56ee3ae997453b8eb93280de0f46e35ef0156aa279e2ba51ceb2f8a8bfd749a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f74edda119ba334ec929b7aa1c0a171b167613b256805029da5f736b379c5687 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0f233b82c8c9ec3c73d790df3579bab07873bd34606957592ad6a439bce03b41 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-79822f39c3810192168f04503c49c256d80421ae36bd712a11618c29290450b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b3ac2199 | sha256-ebfca6de32d9843fd6ba4a2385af111bb1e2040da954858a733b317b74cba99c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b3ac2199 | sha256-61aa89b7ea050b564283b08c54b4c2759ed73e535df35e41d3ef881645c12aa6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ada1f6d2 | sha256-0c7bd15cb5bfc9d8c4a6e7ab0d0b032dfe773fa1fdfdc03fb263bcfab4810786 |
