# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b96b2a97bdbabf2a2491696460b8da77dc242a25c47533759e1ca69d544c781`
- fixture hash: `sha256-32449d86eb6b142eb11e1d76d43e4c37d62e87233bae5b870977e6a064fa97e1`
- score hash: `sha256-ba51c5eb7a651226c87669c00690d55ef0452ebaf2be43b9dbaa7476c8049c4a`
- bundle hash: `sha256-7178d7d229fff02680c3323f43b8ea67a0b6ff20c8ca41c447591e6682124239`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b14dd3575bcaf16e76897e36504d083be01ba320a2077714c9a7749ba84f112 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3a4241524c0c8da7d36f2ef60448ce9012e7b538539a084718a65d8f25eb6edc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b5d429583b5753e954dc8c57fa510279e9af4305336111016c775dae99910398 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a5f31605ab57752ad68579f9a3a6abf51e75031b02d847a836f1a10ed893d83e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa685fc3 | sha256-f50934d43e815c82744b8f7d5e410e5ad374887c51392a6cd12d888089c417d2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa685fc3 | sha256-ebd738e3f74ed4af8ddb26a07a5cb1e0215c6c8b1d17d56d10422bb3d9241f86 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4b52b128 | sha256-18b645298c0a1af532b9ff6ae0b04a878a47ecb4f133b355dba76a96e42f7afd |
