# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30a54314d984e83263bc7ddfcb852ce4d67a835461588938c047eabba74d7daa`
- fixture hash: `sha256-a669f6ac0947e4907b9b5ff0ba78d765904f903d2ac7c540eba1f40434878bd9`
- score hash: `sha256-505c532aeef363328092a4de4dfdcb975552028eff7145c2d0bec7b53d76649a`
- bundle hash: `sha256-ca87b969c5fcfcf73f4bcd20af8dcdd769c608b5fc6e68fd64345c7f13224600`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0ddef39ad20fc1c3136dfb625c29bf78d555d4df3233592558f3107ec01752a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fde3c30c2463990afd47f22fc6394c5a6188a229aecc2346e71489480e6b41a1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-766d7f67e4bdf314ab1a2c93d9c76de41382e187f88478dd8de7705c0d488470 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3d92e7696b5887dc6960ef4cdcb4f18df68b6b8b0ddb696491979badd86c3efa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a4104625 | sha256-9e16594c425418cfd925d745c201ae8b3f1649c6d6dd3bd79e3858f265bbce12 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a4104625 | sha256-55f97dcf97be9fda02eab208bab2bdc0398316d5bb8810d2a5cae18bd5e1b20d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ad25f694 | sha256-fbff568eb4d33c9e9046e599f299fc709d7ea100cdac3d22f23a87e0afbd48d6 |
