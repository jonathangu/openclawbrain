# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c761e47d47331bf575b7d002c131402195e8ac49d688ce355a015b14825d3acc`
- fixture hash: `sha256-128e53eb7404fc5c5e08cc33f7657166db8766e76b0fe254b4c32e80c9220dde`
- score hash: `sha256-0c5ff1b5ebcf9d768598ae1a27e289323567f509eea5078d30e48def1f1a41ab`
- bundle hash: `sha256-159003103b17c14f924153114552d1b1feb37f38f23afc70f1c9431461310580`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1631644a30db1cf3d02f7c72d9e973f8085c5ed6318e74e1e83701e3e901455 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2a1a9cbfb8f4e1eee105c910e583b44c99f73f19a04433a84e84ce28d4348626 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-543a1aa2668cf1e8ea235a9ac6e5325c07bbad65d3ad8968d18acfcf69a426eb |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-75c4894d342f2b9ea7615fb82c4a0da95c15b7013fd38084f23f6fbcd713cb4b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0c595c97 | sha256-29745fd7edd30cd53ecf5b99d619185ef374fc707740486a662fcf56de1b0d3f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0c595c97 | sha256-a26daf21ec27de7ec3abb9885ed4dea846cf44ee2be05e62652c63ed61ab1b4f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ab20958a | sha256-6963295b3fb949a95f080494170b814617e44a87ec022d9e2dcc3a66aa985565 |
