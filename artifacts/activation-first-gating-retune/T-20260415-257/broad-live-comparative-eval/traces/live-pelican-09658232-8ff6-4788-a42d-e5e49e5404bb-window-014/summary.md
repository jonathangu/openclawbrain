# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a2e597e92fe22d4f55c094b8ed54b6a9af6fa4591283d89702e798da892600a7`
- fixture hash: `sha256-622fac4fb2f464038d17b973948a3daa701456585a35960e995213dcda72d3b1`
- score hash: `sha256-c9bb0f0eb36a4220335ca588f22622b6324dc6ecc176b743a3250528142765ac`
- bundle hash: `sha256-7d2aa2e7d42722e4d49081c16e084d7713b2040a83bfc56940dd24e9c7824e14`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-937c7f8b3de6cb0ba567e2def00dcbf253af96c301f9c26d07a7c1aa6375230e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-792d2798abebd8fdb31a71c7a5d4688891f493f49d0ba57fd240a052aa92a1de |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f7dc2d46ec2795f8d8a781515d51791333dc1c28ef908b540a13fa173f760e62 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-dce4cbbbd8375993005513231376ab3b72f9fff2f7ab6cda67cabcc1db06c6e3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1c694696 | sha256-58eb68d3c16433e65d7b663667060f4f61daa3d0788c7573eda7a2b6d59d7136 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1c694696 | sha256-4c73decdb0b3c74f07a3cd6900d1078fca2e0ad156f84f73aa9bdb06522f7b28 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e085285f | sha256-6dde7a6586e625fffe88e6e40e65962e8ff64cb512e2885e8bf6a51e2b625fbc |
