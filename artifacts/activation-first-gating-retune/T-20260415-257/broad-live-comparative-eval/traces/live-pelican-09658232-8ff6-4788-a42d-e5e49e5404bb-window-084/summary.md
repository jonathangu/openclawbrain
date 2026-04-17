# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57d54d463c3d335756b9ef845ab48b21d6ba79bd455096740f5eec6ab5dcf52e`
- fixture hash: `sha256-c975a7548913ddc09f78bdcf8d6f035b2cb79bee5a8fff204c28b6e92be5b531`
- score hash: `sha256-225580974babc27e658c2ad8836852261dfd4e5389614efa581f8761fe13fb5b`
- bundle hash: `sha256-5d19e3972e296380bc1af46d48229a20bb09068f4d936e4c961135f6772f4fb0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f565862470853eb1b48835f5dc58d5e78705c4b54f6971c4806d12966cc7447 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e57f7131a02760b548d644fdd2c768344fc6f74caf33c7d2a15c209224f0dddb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ba66ff741114e23e5804ccd3e7ab450709b4084739231af181083373156ab929 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-1ff62d6efcee6ebb5b0569a84621014629dccc9f28f35645d88d1e8f951e4d8e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-34413cd3 | sha256-9d679374d21f808dd19339a40892a6858aeac0134cc9a19170be8df2375f0880 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-34413cd3 | sha256-8506c9f375eb1110237016876147d84160fc44b9aa7a61b8ce65a4815c3301eb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c01c8afc | sha256-2bfb3bfbf97a97240dba01cc21ce56dc7aef749fd149c78257c9db2fd6281b1c |
