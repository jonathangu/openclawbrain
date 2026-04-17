# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069`
- winner mode: `graph_prior_only`
- trace hash: `sha256-836584be6983eaf5fa6eb8781cac34bfbc9fd538fa4b161ac2d1263fee14146c`
- fixture hash: `sha256-d2752f3a765e793797ebfa0ab38ae1044dcc8b2c28b548d73dcfade2be50b251`
- score hash: `sha256-61ba1192c0d00f5618e59ae06345f9c8a1176caa463d6242af86a9d0125e4605`
- bundle hash: `sha256-0d3978cd92f1940ea13552e141690a6f9e294fb06951e13417a745686efe58a2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c391bbc2363181b72ba8549d9009dc5fb197cd45a7341e37f0fa91e51803c6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-02e6ff8d15ffdfbf2f76a2becc1c08c8a2cda9d647d2041364b7474c55f2203a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f87d527267c6f3d5dc6ede4d41e62ffc9e63a256f476c6890b41da45032b77bf |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7884a34645b4e540850a1ede96e0fce5ba3597dd913e083f8a055b1f608b13a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d5d4d942 | sha256-fdbd6a68383c819a4b3046f0da628095b4a083d49db02a3bfe6da2c159e69253 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d5d4d942 | sha256-3dbbadc2ea08b13f8932a379878cfcd83a213452df10e66d83e0b577261f78f4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-20700ae3 | sha256-575626a77d9d5e9a09d294929f16a971bd8233085377df233799cad220f51973 |
