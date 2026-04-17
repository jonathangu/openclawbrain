# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f896b3d7889710642e066f81d9ef38f09a0375e7c4550a3de44bd42b8be0c728`
- fixture hash: `sha256-02b10f8d55f27089a7a2cdde95f78ea9472dddb1b95943a1431bda089a73cd5e`
- score hash: `sha256-66588ae5295336b4fd365f25019b578c3c954c587613b261cac23d098091235a`
- bundle hash: `sha256-fea5ad99d5170b038c5e41f9d9ec916ec94ad746c00b1bc983a5da05be1be84b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6e466bcaf85a4528dcaf1f22f57a3cde69a22135dbdc628862617cea9e4f77f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b75c40cd1711c556705ad85c60a8bdae0830ec40eded86db2ca0a1ba34e72d35 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5b19e2e503c5c6fe2586a9cb6a72d01e40015a3d8274681aa6afda1b6571acbe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6f00d60a5e4c0ef7a905e390e5f483b54d1d983a9a2f04cce4c9ce5a3a70cedb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7ec622fe | sha256-30d7facbe452958f06618531d0d96c6b82fc6128c7d7edcf30f3ed95c3a12cda |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7ec622fe | sha256-e4e54c895b17258dbecfb9747d89ab6ddf87bbb672a7622ed363dad3f9dd81c5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d3acd0d9 | sha256-8df616b91fa24ec1b9f915fa32a4b54f9eb8b07158d1f238e2f55d56eb209936 |
