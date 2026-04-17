# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-367ba0e9c1765adfcb55faa49a77e3f08a37eaf77c4964ca4eb0f5d706e75deb`
- fixture hash: `sha256-c755dcbf454eec2e6cb44da638da71dca0e7b64e802782c096094c2870f2abfe`
- score hash: `sha256-d6e161ce43179af239a66c61d1206c2718b07fb1d677aa29164a396cdb7fcede`
- bundle hash: `sha256-225959be105015751af17fb5be27fdb419ac4487f3d9f67a95ec4a16d945c672`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df30ae86da7dcb946f187b86df35238c1caa6176c275bd81d1099e4de3972842 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9cda4b3d06e53cfb57db68962f8a5858928aad1369ea371bac101f4d2882fa70 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f8589da2b3af027c47c0ebc1d9dba1dbda9c2dfdad9de1257a1904db317de155 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7d110de692a1fdfd76ef72cce18af542131d427b16ce681c3e61e5f9e013a7f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e74076b6 | sha256-42cfd62389c9d20f24e82f7770064b6ae517f371cf622df848f973cd7d36f0e0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e74076b6 | sha256-3247332a9964ccfdbea4f93a9b8eaa6e95b681ba1606be0925a6b47c588c5488 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-032611ff | sha256-2ab582de5ac77a8efadaf7927e68378feb5202616f9db2bd8b06b0576a5e6e59 |
