# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c46bcb381e0fc3efac0e09438f41834359285b619fa2a6877dd357e64e821071`
- fixture hash: `sha256-1ef85417b722b0f394a6f903af4947a78bca3d01432416a0cb17a206ec104c37`
- score hash: `sha256-727fd2b099872943b102a72f08606dd5d080e064533f6675e2c3693e40c67ef7`
- bundle hash: `sha256-db0c083b0fe37e87f042a9628bc40867a5dce458e4f52b3cfe37801919faffa4`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d30e174da168a75e39ffd3536c03dfe75f2623e328eaa1807c5de3d00819572a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-efa22f2cc2e2243e8303463bef5c06dbb2fa6685399743723e7f718ffb1916e3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8d8d6dab051c9ab3129a617ffaed381975f6574211ef8be26e5012e9df2c8af9 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-1ff0733ba93e1ea06b246c0024e9e6aa32aa6d12786983ec31999b16c6e8448c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1e2d7132 | sha256-d52429859f1218ed135f3264ef110fe85192d7236e22b25d324576269821eac4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1e2d7132 | sha256-ab7baaf63fcacac2b943d5d68a21610ed5681c472bd23149af092aa2b2848262 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-0cfe35d9 | sha256-43929a41885c0abf2ea678b970d7c7b186aaad46112610b344690479652ab32f |
