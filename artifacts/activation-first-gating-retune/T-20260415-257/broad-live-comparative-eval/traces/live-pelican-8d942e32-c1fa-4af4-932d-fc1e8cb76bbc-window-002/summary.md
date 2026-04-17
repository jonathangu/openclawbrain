# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-64a26e8f5e6980d9246d302acab2d34c4e18ddd7be07096a6ca889aa90e2228a`
- fixture hash: `sha256-833c77206f16af416cd188d9e8ee18c5e59708b98a4500bfd6d7d22e62fa078a`
- score hash: `sha256-ac6e4fbbdcccbac07cc5d8c5790206e56c79f529e2938550ba37bf04d0e00735`
- bundle hash: `sha256-a265fc5df6e1354876315d6f8159fd061eb29e193cdd1e733d5b8d22d023626c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-947e512b0a82ec6d517ce602229a8e508d29ae58b836a4631a42b14c828dead3 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7cde3c944a9b076a5a74a3ac6a76ebf7ec7b52beb9d6953f27de8f185bc4ca38 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-68f5927a9760370eb3fda81db8fea684cb6f1f698719b6e1487d66b024f77ad1 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-ac109058822072e126fe049ebd4e854c54b5cde2f8a1f2048719d8486e4513f6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3e095174 | sha256-6cc0e37588a2ad5b8d716c0063eee6e94250c2b0c25f385ed328c9749b661e85 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3e095174 | sha256-0609bfd2557347c56b429ecbfd88d7a257a2b9ec74b4da6ff7f0686fd4449768 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-39059d2b | sha256-cb10bb2d0a02d91ffd9e2980aaaaa63fd30c3831ff14c25028ca72bdb09e4a34 |
