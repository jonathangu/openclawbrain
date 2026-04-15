# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc7aa6c27637d299d6eae706b4fc67a2a2a7b4de77c818a562317ac57ca7ac6f`
- fixture hash: `sha256-3d9a8c7638fdfa743ac7a63700e6bcceed5b6728eed1bfa78f1b2db0ab28c6de`
- score hash: `sha256-fb98adafe1de61d439b3dc5a03003a0e3d1e50dfcae3594eb33d70ae29211a74`
- bundle hash: `sha256-8ca4a7bc4e846bb99f25d932bd5da1016ebfb5fe334d9bc624b91976a84d531b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6474375f5bcb6a5860753785382ca496af4bf19e7ca31262302583c0776eda20 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-44b430e09933452cf412552017d8294ed72ef673267b2f52e4ee10473698a059 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29f521d3fa1ad432ef0192c6e09d19b1294bf708f3a29da4428f54097e07ea6d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b0fa049e5f4d21a36158a5e5a9228067ccf2f1341eb71896e1025f9f47185d49 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-327f3984 | sha256-630ce0ddebb2bef847add57b963c7dc2a1767be4278f32d8e93f480efb4d52c8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-327f3984 | sha256-bc5d0f5d80201e77f7bad0b512420d8b5d1be95d34dbc7691daf3e35332eda51 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-327f3984 | sha256-630ce0ddebb2bef847add57b963c7dc2a1767be4278f32d8e93f480efb4d52c8 |
