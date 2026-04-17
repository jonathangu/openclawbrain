# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84b9a4843680de911479c2420a8592984c3d84b3d54d06debdc96d5c918ea030`
- fixture hash: `sha256-be373dad3e692162d5000f12580f9371232c68a9b0f09d3136130b3fe2a640e9`
- score hash: `sha256-141ee5f525b327d121516c718f53c2ada631851609ef1a0633050c1ab28079df`
- bundle hash: `sha256-9ff582e632b875ca70123f4177d0b836a4b8e79cf2bff854a03f2897b7bc4f9d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-34864147a65f338d5fe87baff27e70ea8462feed84ac2fbd4644ab5e3e006364 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-803896484fd5e0f7e81318e4c9e32b9061fffc31765da7b2663b2eaa0efc9094 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-041eef5b6e9f5e6fe86963530dc9bec0dcdb2bf2d938f1b86c8c9231ccdad9bc |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-483e0ca2cfb50acd935e72fab1164f7961a98ffd30a46248ab0fa6ba9b041942 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-95898a8d | sha256-7e1c8e406b61114358df2971d660326ffd102ed89382a99ee538494288a25b46 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-95898a8d | sha256-ef847cce758f616327863adfc0e8dbe4de2c070807c4b66f9bf31b52406d0fab |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-48dfb8d0 | sha256-acf8bd08a3a9ebb735ccd0725d1a4946011aee2bee863ac5495812581a6d097b |
