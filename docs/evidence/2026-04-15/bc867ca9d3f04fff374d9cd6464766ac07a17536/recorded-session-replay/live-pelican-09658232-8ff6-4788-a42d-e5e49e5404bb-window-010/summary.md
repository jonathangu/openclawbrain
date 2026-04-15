# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd69cb2bd54df203880c8fab0fa4c855341f06ecfb9d6ec642144558419aa71a`
- fixture hash: `sha256-f29630fbd2f41b8d395fae06865eb7778e00433b1298788381332e0703a42702`
- score hash: `sha256-6e36ad63345572c91e115b076217b766a64b0c2515676d7bbc965622000b2a62`
- bundle hash: `sha256-f53e2f4ace2b2dad1cc61b3bdd186e377e511d84c40adf7c8e3c6d2e5a69c5e8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80e65d944c503b7cf482a4ac157c70bd9810fcdc3cd3dc77c36042f87f3356ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-40daf4e66648428f8177c071389f16124a44dd5623d6b41dd96d59ee17c4cd12 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-544406d823db9c991a46ab4f18243b96aa5fd4ff3b02604b89629f639e800a02 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a9b9df600127402bc1734fb13a07b3db729d099b6f91f585c46ab3019f51cb85 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f0c1bf9e | sha256-437a70c5e2e44ba988c028a7eea83cfd89b67b4f353c3470a8617299a0b31173 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f0c1bf9e | sha256-26f6b6fdf951afbcd8c0952af6e12c77561e50ed8cfd9c835366e18bbd823f7b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f0c1bf9e | sha256-437a70c5e2e44ba988c028a7eea83cfd89b67b4f353c3470a8617299a0b31173 |
