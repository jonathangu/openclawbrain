# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-caf66a8ae0f50cd2a6d3d3cfd5e877c0e6e80108a88f78a99bf9d0af8e14c2ff`
- fixture hash: `sha256-d0e5149f0ea8ad48a690b98cf321460b1ba9a083ea4dc63f8728a3baa728b105`
- score hash: `sha256-5436c0623c46f1036c7307babe83a91db58e6284c956ae22b12e7f9a98023b23`
- bundle hash: `sha256-b7bc0e1d1f98f4b6f0a8e90e227938c51f6243af6e2a69c5c40783abf1c1f6d3`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9082b7d02459eb2c821fe90896dfa510257befaff2be31c5d25dfe86c62c130d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-27514c0dfae38ba568b3aac7c9fcf45570ebb8fc43c37c595b23c5866039d9ed |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-a6eb042994f06485fc804b480ee887de145e3b50604b74223baa7b32be56f9d9 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-7bf7fbaaa156d569e613809b555f373f1a214a11808d14db5a8b91ebb232c105 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-60ea22cb | sha256-5848cf0e94b5ea11498cbf078131cceff624ca0b8bf01c7696cd7e03eb067466 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-60ea22cb | sha256-7d04a97729d1872b10e9159f33194512685f8c0d14944e0766a5f904bc889282 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-534b772c | sha256-1f26d4cce7d42cc3cd11545415cf4a0ffda0760e899d69cb598f69f6f40707ad |
