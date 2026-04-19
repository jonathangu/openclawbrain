# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069`
- winner mode: `graph_prior_only`
- trace hash: `sha256-836584be6983eaf5fa6eb8781cac34bfbc9fd538fa4b161ac2d1263fee14146c`
- fixture hash: `sha256-d2752f3a765e793797ebfa0ab38ae1044dcc8b2c28b548d73dcfade2be50b251`
- score hash: `sha256-e9f044e85830bd467f376260e4e205ad6537a3a54764ac136d2e94e2e0c2dad0`
- bundle hash: `sha256-0671fbe13f84e239a942ee21cb0678043786ef66612efbd5576c754406e09e47`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c391bbc2363181b72ba8549d9009dc5fb197cd45a7341e37f0fa91e51803c6d |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-856de7f7523c2f54910ac9616c7186fd012011052c71c8a3ba7738e4cbc59027 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-9357d26791b59179170480925cef7d228fde6e850f31fb983a8e3aabfd74d08a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6915b8647ff0b03fecf93071dd687e105446290b5743dd42a152ebc28eeb06c1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-9ea1158c | sha256-e597fd6d9b8e6354914b9bdd0139f657cfb1f969b40c66d0b45d7703633c66d1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-9ea1158c | sha256-cae89e49669ee7d6454cbd17c6e5bbab5c11b8f6cdbaabe8cd1afaeb68a95bfe |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9ea1158c | sha256-e597fd6d9b8e6354914b9bdd0139f657cfb1f969b40c66d0b45d7703633c66d1 |
