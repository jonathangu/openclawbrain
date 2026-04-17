# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba20557b6de7502d32e3b83fc90ffd6f8ac19ac17c3a2b682f0895bf8bb69c7c`
- fixture hash: `sha256-024c24fcc3f69f4d62a086b795f3d8c9e3625b36454d26d1e235e1664a651060`
- score hash: `sha256-9aa825faecd9020fe11e9f92c2f2417dee4574f6ca5cceb22956bb541a9e813b`
- bundle hash: `sha256-91c95fa5a2666de941d540284dd43c99cd26cd88c41228dfbba4a6921e27e1df`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91ace2bc1ee23370cade1ee9720612db55ace83c09692395cb273562a40c2beb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-10b3c6072b50e96793e2865bda9e428915b06d341d9449e3f08b042d33cd226d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9fdec135c694431baa6fc566df52d7bcc207d0c45455afb0752117bcd36e71f3 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-3b9af050054af928af8560ae3a43580d66b8f0735cb82fbd15ff114bc483fb50 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d3ff4b18 | sha256-75e56a737fc5e16993ae85cf588dbd2c19efa140bdcbd56ea42d38b478bda24e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d3ff4b18 | sha256-ef04114fa109757b47353d220eaa79267173ba1a4d70bf787e245180874079aa |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-42522541 | sha256-eba3e4c0ff93b8acd9aff74cc4eafc2c7a9bd4d133c0b1c7b172c1b7cc6c9fed |
