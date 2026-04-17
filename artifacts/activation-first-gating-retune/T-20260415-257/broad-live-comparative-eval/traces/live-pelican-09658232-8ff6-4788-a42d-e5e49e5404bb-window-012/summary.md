# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50bf6e94091f3556fa81577b5a708e4425c4e417b6705bd10df603b7966e593f`
- fixture hash: `sha256-18f79e8bc777ea9555de29a01a8501d21c8ddf1c9ea32bbf589d49b4f4a3aaeb`
- score hash: `sha256-d8ea35af40df72c51f2ff2b3242801767b8db7bf9ebacdf9630ea2b3a45fa6da`
- bundle hash: `sha256-4691a42b6d688772cfbeaf5aa55063aaff887590139d18e949434636d00b5daa`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12a31c86174cd8cb94081745ed9b01e9e8efd75760d60dfa60b0b81778821ef5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc9e47eaedb7858c606db2a894fb46d017bb1c12447ff80f7bbb998ff209117 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b3e3d473230ea271fdf6bae1bee613259f98bc544c1c2b5d2f5d09a4613c90c5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-457af70cb4554963d04c8674493673e1ab534da51303c51d93f13f8770690b44 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ebabd1b2 | sha256-c514fde4ffe1e2ca204b51a8d588ce8333b75a87d894b3bc7225a3c4fbdc7187 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ebabd1b2 | sha256-39098bdf4e968447a3029efea657cb6065df788719d2eb658c7b2834ed30a041 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ba46bef7 | sha256-74b7339a0c482f9ecaccd3d63d121f0c1b644e3afb3ecc06634082b93fe6c95c |
