# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8eced4262f5a642239299c7c899085a7bd53ad7880d03357a10326803fe33aa8`
- fixture hash: `sha256-5aa5748a68c006cb4152d6b9766d43523c43872689382d99e9608f0fedb263a8`
- score hash: `sha256-b91bc04590abdb861272741d052a3deb09fb3a893e67420af42e38f279b4a26b`
- bundle hash: `sha256-deb0f4ad5ff92572d7467640c1f10cc66c6ed8ec2eb7c00ff10dc5592023eeca`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86ead80920d9422dc3144931f0210740c8474d5a0351518c55316e7dfbfbffe7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6852b6292884673f642ea76c266f3966f1f34a8fac29b5b1cc23dd0d7e185c34 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b01416d44917d95fc81efbc30551fd0ef0b68cf6ba1e5ae85babe06e1a238bdd |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c48f79e417683fcf1fb5d752d476e9a1fed1fd89d5fe054e6ba251ff474df857 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-40373619 | sha256-01ebff896d86f05075fe05fa2b67cd4204bdf53f7f29641f3d5137bfdd45aad8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-40373619 | sha256-fccabdf1f9799763e4b23267466378b34a8dc2aa6378e7812d2129f93d64bbdb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3db8740a | sha256-e205c618e79973cf1d5eecad0b308f14ff1c2006728a58f0a250606cf23e3135 |
