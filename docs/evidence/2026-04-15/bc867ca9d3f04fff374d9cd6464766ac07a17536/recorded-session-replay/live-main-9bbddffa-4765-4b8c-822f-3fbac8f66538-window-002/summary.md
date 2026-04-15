# Recorded Session Replay Proof Bundle

- trace id: `live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d82a353da8dcdcb26be266a0d33a583d117e1f2075d582930b7ead32e3d715fb`
- fixture hash: `sha256-3e1329cac030395635745dbbbafec0f460454aaccbb63b26f155aed0ae65e7c6`
- score hash: `sha256-23dab062268ad10a8cace995c38594ce171aee0f946d19918218a2471b01bce7`
- bundle hash: `sha256-ea00070f520c6abf2d227100dbe05ea27e9d1f8faf21be395a1b62b49fc33e1e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38cc6030cda1cec211011cfcfcb3fe3c0763917e1cb1cee36ef6155b409ff4f0 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf29636c51fbc0d684f0013ef924d3764df89974ade1ccad968863b101f24d50 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-5d7d9b09c5fcd4ec10fe90dfefc4255227a61df5fa732741d3f6b57ece7a5f8e |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-fd5a4d8ea9d81a1685b1115c05f6fee738aca966331e747894e839e7fec346ab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-c4b34300 | sha256-bc226a2605d2f8aad8711051dfcb62f3d674ac4b34fd14dff498f987b38f8b19 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-c4b34300 | sha256-1531ae33220064a4e6b3ab14afddfc72298cf2de2ae522579f197e98f62d8cdd |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-c4b34300 | sha256-bc226a2605d2f8aad8711051dfcb62f3d674ac4b34fd14dff498f987b38f8b19 |
