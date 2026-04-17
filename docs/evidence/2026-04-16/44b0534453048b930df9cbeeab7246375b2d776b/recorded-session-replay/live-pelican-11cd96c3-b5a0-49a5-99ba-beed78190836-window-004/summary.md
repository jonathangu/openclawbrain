# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7ce9d77f8d5b34f5d4a2ff238035837b9a17936c8718ddcd44e0135af5ed67b2`
- fixture hash: `sha256-b278e1b6b555771ff403bacda1c9f56aa4593110af14f3b45502af98316b55cf`
- score hash: `sha256-44274d30357e11057cb0591eb5ad6d8d56be3b7ad761a78c244eea42678f94c9`
- bundle hash: `sha256-732ba8cfeb796ead0b16bba7e66d30c742ad574c1061a9b6194052d728e0261c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1fde92dc9f75c3ed7b5cdbc92af57a8fdea90f988cee9df5a6592eb109fc517c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ebec909228cc5058a6f4a1e1a5bfdc15433843fe7ec9ff3f9186c463f39cd181 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fff703a6471db776dc208ed8d87efe04ca171840d73fd7d4442e924b5229f573 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b3ce6d503b9afe79b112f348a65d8a2a8225cc0e65dd61fceb97a51cd2f8af65 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-07572278 | sha256-ecb70d78fc878095b261634b61b0dd8474a179833cc407d893e6826217384322 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-07572278 | sha256-55c427e97449204df6961a6872af1eca75616694c4cfe3f075a2fef2109af8aa |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ce4d6d45 | sha256-900d935c695e0ff6e3c0d543b819dbe62e1f8ea5f5c97e7257944617a2c90b6e |
