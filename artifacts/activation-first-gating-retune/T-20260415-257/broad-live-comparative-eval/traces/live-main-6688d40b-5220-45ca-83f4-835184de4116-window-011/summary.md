# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-850bdaf9d44fc5882480e1fdabd688dfe420007059248f68b1bfcdb177c8d991`
- fixture hash: `sha256-01700f6ae7fa9661baee2d1698232fbeb6cde54e151f8324fe1800456806d50b`
- score hash: `sha256-841af0b97fe010b291170c89d4c2c2bfc18a109d9b9b43ae90ace88990c897d0`
- bundle hash: `sha256-a454fcff645bf9ac09b2aaf0f8d4922ff8d18897373e6cb29de175f9e4798b1c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-db5ed18ce329dbd2d8fbf4381eae760a575d1622b5dffa25a4d7dabdc4b4d367 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-af49f778dbf9d1f45d99c637890b676fc002bfb4fb871d875f6664a5dd47cc81 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1284c1d095f8118a6171b019b41e09f149c4fd8b3a7f2c8853ad3900ed348791 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c01c015eb191b2e1ff2a13062cc32153b195511d4881c56922f8aab916fc265f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-403a88bf | sha256-9bf408de4a03228e8b5fd98559244cdf225fe760a4308a28eadbcf8c2d4cb365 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-403a88bf | sha256-9b8082df35ab798b88f25b4e47341e681a3beff3db6bf90805801899126a85a2 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d43627da | sha256-e857e3845261edec2c10552ca08df9f189cb132f68e58cddfab4943de004f516 |
