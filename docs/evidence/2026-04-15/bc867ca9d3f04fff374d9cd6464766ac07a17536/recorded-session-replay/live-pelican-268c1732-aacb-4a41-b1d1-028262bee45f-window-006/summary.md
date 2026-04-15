# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fc1409b104d617856751474f01593056b66d1b2ca492e8f5dd879839efd10f66`
- fixture hash: `sha256-8310747322d42de0fb2d06597a429aa5eb75a2026f88cf3e458dadef80911084`
- score hash: `sha256-890f88d0412130edc0aabde92d49841051145fa9d2dae2547af58bfb426c6f6e`
- bundle hash: `sha256-e61aa909f4fb624b9caf78bace979e654f36601e2bc96ff51b10a5e78f5c531d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fcb97b42b2b441ec8190e1bb06fb82b8bdd1457d8fd6d8d105b2684066c5870 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6bd36eaaf686be8491990e12ce349ce7aeb5524c01c268e670e634a74608ad7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b8f60c429e0a8448149d8e4db79a62af034d81f09d88c8bf2ce29f7fdf48d1c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-52b13044ee9d8e7d3e1c72de50a375a6eb0bbf5b68cc3bd5fc8b5b7c83a778f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-de01439e | sha256-d556990a8571c0904428bd19e5a1187dbd528ba793b57acf9815965198d2e2ca |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-de01439e | sha256-bfecf0bbd64897c5105c9af24a16ddea1e974b1b42bbe48060e795a38a4199b6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-de01439e | sha256-d556990a8571c0904428bd19e5a1187dbd528ba793b57acf9815965198d2e2ca |
