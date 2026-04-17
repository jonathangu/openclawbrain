# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e8abe8bd791e7d6cf823eab880acb642edafbee61d1547309c32e0509f5a12fd`
- fixture hash: `sha256-55ffe1baff231052090ba7af248a8c8c581b0ed9688d4757d7043a08a2fcb4de`
- score hash: `sha256-77746e2ccdb7711941f1d9ab035785758fc12dc681ed6daec9a7bf5811391acc`
- bundle hash: `sha256-09ee308d8f9ae00a5048ce8c131630009ceec42ca49890de85e19ab8885b9752`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1da03994a3e5454931ba1a5c62fc1691a06d32d29326ec5baedfa4f4b490d130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c3da4d71da251a2aa7c3a065849cbad2dc184e4952b04585882fdf90c4fd7f5e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b83b9216651ca6ee2bc34e273c744b4148f7e391cac038830f0ea2bf941accab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d7a4546ad5351c94eedd92da0bd7d0b51818148265e65a59ea2e36e3bda729c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-20661373 | sha256-473df550930a4d46fad25baab68dc6bf9f491d630b1edaf409beb5b5bb8330fa |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-20661373 | sha256-45d993ebfc2c2e484730ca14e8f1f694eaa71cdbb8f3481df876bd04daf4ac2d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-334dfbea | sha256-268672de155e9230f61893718a3f5f33bd865f5f8e4569ed95be7d1a2635ce78 |
