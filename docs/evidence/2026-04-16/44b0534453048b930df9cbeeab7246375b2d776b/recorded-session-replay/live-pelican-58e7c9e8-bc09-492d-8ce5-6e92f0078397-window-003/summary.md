# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-529373cf8f7314054ee5a9938b5133a303e70a9153c03b373cae4ff852f394c7`
- fixture hash: `sha256-c16690eb3752325552dd8dd957f6a57c852c3d697d1ce7463c9556556d92ca19`
- score hash: `sha256-9b87ae171cd051389ed75938a6a813a72e7ef7eefbe5a288d3480a364641176c`
- bundle hash: `sha256-3f989617db422fcfc52278389507664f4ab31d0be410ce15e6c73623b1d2f09e`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9641da18215ca2d07fc313a19aa471e30d85d3a5754d470ceff969f5080d786d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33cbc76dd71c274e7e7564f75bf198ab54cf4a4c702aca90ec4992720001d9c4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ecf51c442ef88feff6383f2f9a0d6d69a27dc0be69783e264cddee27a18b7c87 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-224ed2cb137f3a574ec28df4aaf5281b361e071125b8230e2aa7fa3ecf51895e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1dfe9920 | sha256-65d40911eebf84176a82f1ea933d56f89eb756209e76156e77d69ecea1ebc4b5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1dfe9920 | sha256-3209fb1cd8d13a2ad95d9c35b308d1c1593d9f39bd545d9a8c595490733090ac |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cfbe298b | sha256-db8a0849da488d376bb64a722cbb38ecc0a35487bddcc2f0b673de533eecb972 |
