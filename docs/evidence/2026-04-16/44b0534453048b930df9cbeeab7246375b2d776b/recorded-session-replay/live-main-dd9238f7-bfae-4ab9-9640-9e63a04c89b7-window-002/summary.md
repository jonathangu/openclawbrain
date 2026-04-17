# Recorded Session Replay Proof Bundle

- trace id: `live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f45e438e0f99f9d56b4e0ce3ef341383c4f2368651efec5583a2c7447c8a5e0`
- fixture hash: `sha256-24e221e1cec238f614a332fafbde124000574c7f4eca983f394d512d73646f16`
- score hash: `sha256-4762abca830139a714fecc81d2b99536982863a502898ea047c380b4d6266e22`
- bundle hash: `sha256-673b8ee4db6906c52155ffacd3837f43edfcf90fbd1f0d124289669fb12257dd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5624a20b92c6a5c4c5d269dbfed46d621fb3009b7407cdb61d3d2abad216a892 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-783ebeec14645e9662f8a30c93781cf779ad6b0509df9c4663b5a10dab8181cc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8a522da3376b3e7b092f76e2948cfa5b7f8ad2f2d921dadb8e918ea2343a30ec |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4f3e80ba81b07b093b4e32126d792b78df96e4c1d75db97a4a9b63ac33867cfc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f1c2dc3c | sha256-c05dfc76da33c66ddfc82ab68c73b41bb230583c0fc74bc0a8a4debaac54db15 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f1c2dc3c | sha256-56155d900883987b3496c04d11c413ec8aadab5145cd4be72f01914b1f0f8f8f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-df4c628b | sha256-6675bbc9c6d0edcf0818dea826be68cb885af42eb6bed6b0d5942f52d5307ab3 |
