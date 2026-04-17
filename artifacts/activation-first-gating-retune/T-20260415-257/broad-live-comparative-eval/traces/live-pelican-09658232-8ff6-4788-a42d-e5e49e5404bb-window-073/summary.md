# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4344938604e067860a8ae5cde1fca1ccd4f50c2742543e1ed5dbbab203e23d74`
- fixture hash: `sha256-835a3394dac9a8b8023e71ca801b0ad86f7853cc9e826a2ddbfdbf3c56dd351e`
- score hash: `sha256-be0c1359f1271f7f530bfdfa21f9c260d192df771184963673f39ce6373c6c81`
- bundle hash: `sha256-5bda8a325e2dfb9ad780a5560570eb6f8cb5ef3bf39b6b7b5520238d0260dc5a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-95f6b633b00bd779574dbd24baa772f0fb4eebc8350ac2c13ddc54230525a7fa |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-22cb115b75966ca81c5b041a85862c97657b2b298bf6be95e52ae61d6f7ad1f0 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-abc92c5f176d152e6ed9b6df9df0a322edde6e096bc306e45635532efc7c378e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-bd1ba16293bf9dd151bdf7a35fc52baf7c535128a9da9b820a527e292ea795bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-07a85226 | sha256-afeea9293a6b4d3f7dd80e0f5f7f239cb2488803a550c217843f693e0e04e2dd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-07a85226 | sha256-5d83dc590603b6b0ca51dd0b56d967a1f73e1c5494897f37a11fcadb19bda267 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-50844de5 | sha256-806d07ad6e58fe47d683833e3b44d2dbae587e39980cead41f4cfb49c2fbb089 |
