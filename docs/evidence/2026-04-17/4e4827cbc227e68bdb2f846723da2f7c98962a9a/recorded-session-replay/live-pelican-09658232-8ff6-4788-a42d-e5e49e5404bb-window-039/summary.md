# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30a54314d984e83263bc7ddfcb852ce4d67a835461588938c047eabba74d7daa`
- fixture hash: `sha256-a669f6ac0947e4907b9b5ff0ba78d765904f903d2ac7c540eba1f40434878bd9`
- score hash: `sha256-bb2a1a17df3e6c527eee4e7e545280fe5f961cb3891a7e5e7067dcad54ca8959`
- bundle hash: `sha256-4d1e3fa44742aacadb3653476e8950b1af5bd47669a63d70259154042871fdd4`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2d8401842a07565f596cd646681e80821e78735bd194d9df7b1593551f509921 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-120c2aabb50be320ba6f9abdb4c0a92b56146ee09d54bbf8db77a1dc3be0089c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6d6a957ad6843bc03770ced68dec0138bb67dcd589c536a3692a5a88eebf9889 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-96c701ba | sha256-ebddc5dd038e0d9ff7cb3780579fdd0b1937e8d7a46ca74022ba38204d660d43 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-96c701ba | sha256-9ba19a5b4d10acc98199d1e4621f67963d9560d25f0f67cb4d8d40cd81b6ae01 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9fdcb229 | sha256-6da68b4294c05295b3b6c84e3a4a2f093ba8826e80786f79df00afd3be1b7bb3 |
