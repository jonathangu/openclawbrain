# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba20557b6de7502d32e3b83fc90ffd6f8ac19ac17c3a2b682f0895bf8bb69c7c`
- fixture hash: `sha256-024c24fcc3f69f4d62a086b795f3d8c9e3625b36454d26d1e235e1664a651060`
- score hash: `sha256-28f26cae824b426c3a2cb4a2506dbf716e3c3a47f57bc6458b90570ad049b551`
- bundle hash: `sha256-b002a9df5d07d26782f3cd8b3c0cd6894e62bc58338bd199fdf83d5f7da45879`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91ace2bc1ee23370cade1ee9720612db55ace83c09692395cb273562a40c2beb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-add67c7930caa98086b7b0454ddbc797de616a73eb71704f77067f3824216446 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-23ec849cd8efcdc987f17b505a79c20b10d0ff28831c852842b3c601693cef78 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-fa78ec4a884838b130edd18edd09308aa5d6cb9ffd0eda3600a86142ada00b9e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-aa5da7d7 | sha256-67a3890ade037032bb117c775b1536d50ebb2fc2105d3e92dc2e0169fdab1b52 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-aa5da7d7 | sha256-eb6f5ab10115853c54198ff27eff953fb6a98446270c790d93c39b33f5effc30 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-18b08200 | sha256-07d3d31949e61c8bd90bccfba354ee9f4509944acfd673bd3a4c267bf87e7196 |
