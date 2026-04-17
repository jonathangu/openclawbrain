# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11f89d0770e58c74a32e0ac08329b409440ac220cb647ec446567aadc15cbdd6`
- fixture hash: `sha256-7793c2d77fac055a1c7c47c9d026a76a01511a45ccb17bbe5db49943de3d0ea4`
- score hash: `sha256-17c2044d3a87c0bf6edcc6734c7350f4d7545ecff25b30adef2b2307b3337966`
- bundle hash: `sha256-744a8189698942ff67e6814e0c044930d241b4de6ae3f3c51088b3cf953ed846`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3297dca5d83084645cc80493377a366cf545c5142415159b972c4f8430720ab7 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d09d5623142393b10ec6cb8b5287ead2d57f9a56f0255857d9ddd7e01d7a3ac2 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6a6aa254c0e0f9bd9b47cf59f45573c96144b770cce620ff9c703d1b2d1aa90e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5e34ce5321f1b987cdccfb0d7e8e493715c043b8ed2a9629ab3ad80dcf27301d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-ae530384 | sha256-8f0134c9c1c049e247a6981e7715ec984ebbea2db6b30a0af10805e219d5609b |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-ae530384 | sha256-51557f23fcea74e738212ed2470b3939d7ab07454ec45c3f4313af8242b3ef99 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-09c3e99d | sha256-a5243b40c331b97872d6dc1e1ebbd2119a7ac8eec3e7f7b243768bd3ab7111d7 |
