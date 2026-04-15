# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ea52ae43f846ec5905bcf93ba8cbe911779b358315c7c925dfcb3bb8d88c42d`
- fixture hash: `sha256-5ca126c8a28da22685d19c74b8dc7e5cb0bac37c0b916d2162c68f83275f6394`
- score hash: `sha256-19e8dd1a1fbb611cf3506ebe22a72edc807ce125430344ca536672efd323025e`
- bundle hash: `sha256-9c7b4bcb40461dd9b258b76627491ddd13ed9888b9103e50812fb7bab14a6ac7`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ee8e31cf8824d425cda16aa09e9361fc2028a17b7c4fcbcc21c2fa64f147edf |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16fd5ba876114ccf90fda85ae57706f2d102e40216c57e38572fc08d3d8e3683 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0f28b832f7fd2203f7545d0fc57c9e80f9b0492dd23a88b3e36797755a6dd27c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c3064b11e9db3ee1cadb40976beb0e21e2fe24bdc716979cf1b94c78de0a5d95 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1f915cd2 | sha256-9da889c2befbebd1cef55188e43571941ac96b63cab735a94349a7c29ec985a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1f915cd2 | sha256-f88644717477c7e690946b8dc8b6d558852dd209faf80aa7c640b69eaba4e11b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1f915cd2 | sha256-9da889c2befbebd1cef55188e43571941ac96b63cab735a94349a7c29ec985a4 |
