# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ed2f9f31e28bc4c542ba13fc0a4ccba3e6b6e5db3982235d09f16d62242d7c5e`
- fixture hash: `sha256-c571aef0c0ac7b60f97a81ecefc88f95d1024f6a761836a503482febdda1b1eb`
- score hash: `sha256-7196032979565c423c3779773f43d55031a5b351a696bc9758b1ef54adbac3f7`
- bundle hash: `sha256-184e347d0c894697139809dc81efce493e417327a9f14e78b16db2e97e0bf460`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4a371a11e3f0400310e154f8ea3c13a532ee5c397c446eff3697fe01cbdc026c |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-273735477628b8e27a8b2db8ccfcf47a1235d6fc599f5873aa5a0d467dbff77a |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-3d26e721531c7e67de8b61e72ec2343449f2972986f83ed5efdae796de3d1eec |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4510d335b4d4e39ab64b996aec8c016b50ef591625f8d80c3368e008e5ef8672 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-30fa17c9 | sha256-ba02cbdc6bcfc06c602c984475b11b3af0003966977f25cf32480e110a2ba9ac |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-30fa17c9 | sha256-078858d3ed4402d79b8d9f9c77a5163fbac65f112b440bc88ecc80f166090cd5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9a63808c | sha256-17e703ffb0c23b5ef3e9044bc2a3e6f861626526764bf2e555533bdea5472d84 |
