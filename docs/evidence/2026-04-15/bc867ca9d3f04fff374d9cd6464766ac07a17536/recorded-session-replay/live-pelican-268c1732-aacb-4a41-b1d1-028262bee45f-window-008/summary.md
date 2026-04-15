# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-821174c9e8199055ec82b211fb2d09a993a3651df37b6c8cbf4707f78e0465ca`
- fixture hash: `sha256-bcd47938b5aaf944e8ae47149b98261af7f5e6b78cefab4ab8c21ef4d0f8288c`
- score hash: `sha256-b74407b8e97ee1dc0ff7862e312b882532a45213fc5893525b2d8df192dff77c`
- bundle hash: `sha256-0d2d6df9ae098ca077dfc0901d9168f967c581e51f16d4f45668d151e99de53f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46bef670c7b3d0d500dbccdbaeb44127bcccbda5425d78ea64b9256410c95a9e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-85a87d337c512d3c659d872c9c60e621a6fa7868fa179e4fe9c29684f25df377 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ae023c9b3652971ea076ccaab172bd607c97512564069c6d9e58fe83e539c1a2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-57bafe338c1206202082d31c232dc03bfa3e50bd845855937d290daa5b365410 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a3da5e1d | sha256-7f3d806111c7a1d465e0411c93f78ee9382782084732c235102cea647fdc70b8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a3da5e1d | sha256-d7d968e148b497976ee443f46f61e1a242619c86f16b29e1a952f96399720cbb |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a3da5e1d | sha256-7f3d806111c7a1d465e0411c93f78ee9382782084732c235102cea647fdc70b8 |
