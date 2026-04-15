# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-64a26e8f5e6980d9246d302acab2d34c4e18ddd7be07096a6ca889aa90e2228a`
- fixture hash: `sha256-833c77206f16af416cd188d9e8ee18c5e59708b98a4500bfd6d7d22e62fa078a`
- score hash: `sha256-c7a93547db14456a62972ba3cfaaae0dcfd431e396185b3ba4e1519fd2439b78`
- bundle hash: `sha256-dd72e99b3e1661379ce2a645cd24361c06ab215b654b30838174795e0aac61cd`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1d89dba1920d7f42314354d372e0f0a9d9a34b2ebb3c8fc0656eb2f6c41ee2e4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fa1afa2013d0fa4249da565f9df0bcdafd9c18a638bdc100c30c15c6fd4e4eb4 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-46aeade905891023d54c3080737005881b8b3dcf084be77e34a21a231cef21c4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-18cd2b33 | sha256-01ca9a7359723663048d8c477fafff903f547304f036c718e6d5329727cc926f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-18cd2b33 | sha256-602cd2c9303901ee17130cf2ce34472bfa085c5cc8dd3a842a901c50ae39602d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-18cd2b33 | sha256-01ca9a7359723663048d8c477fafff903f547304f036c718e6d5329727cc926f |
