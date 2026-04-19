# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e57ec68d935737d06aef21380bbe661fdacb97877b7cf7bc1a4c9589d64ac9ac`
- fixture hash: `sha256-2dd65164f5290b5ce39a35df1037a21b22351c6c824fd1679b3dae413abe2583`
- score hash: `sha256-fcfe0b98657baf25fd805649a70e2e76fa5119f37dc8cc96d87e6b8046575f21`
- bundle hash: `sha256-014b7a38aa8656bfea52a0ebb534732a4605d34594804fe4be6d856219369e22`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d76563baad54aabf2052fe083a937d82e3fb9d0d74cdb60aea31ae988aaf3b8f |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f68b441ab2848e74432e19ce82f8fbc702528f8d63d885bcaafba5695c7ad5a5 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-eed6026367ad1cb6421200ff87e43ce5b72b445f839ff9d2295dd00dd3592313 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e07854d906a44892f6cfa995cf250c8ff248c944c9b4978744cdf53749738901 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf0a1907 | sha256-8dc0dcc00255f16ab871e210343b5a81af24830f3621e17704dc7546d3e7be8c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf0a1907 | sha256-bf6a527a73ae5222bfffb6ac9c357dd33c442ae34013767dc1166209d2d9328a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf0a1907 | sha256-6e1a1354721d66b28a22e9cdd3d8a2b3f0bb7bf15094e047c493c79ed3e513e2 |
