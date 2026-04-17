# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4eec8363dbb342027ba768d4c007a5f6bb26136392615c736348968a0e88b605`
- fixture hash: `sha256-dd05986903c2d3b37c7fdf438fab0c6737ffb4bb4a24103981b21ff15a79f4c7`
- score hash: `sha256-ca383356bb66e3e455d52fac6915b81b4a1ef591e88f99aadffda5e52ac01459`
- bundle hash: `sha256-55f47bf4f75daf95c2987b7394184c560f717f835acbd1e7f578ec6e6af4d3a8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1952d2df118914f878f9222de732246d88461cd2b6a05bddbf4ef8392b473715 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-acf91434060dcfc750852961d6315014c27d49edce59e529439e6f2d8001aa6c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d252fa6c5ad0b14c5f57b18c26f9f40a13649f3caa180e859585bc47ed05d225 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-db7b35dd0e5b8f1c4ae02d29cef1de5622e4bccb539b49b0f1d46c6b4782c1f1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-16dbcb1e | sha256-887e912ff7e05ce1f4eb337e1f039ce1fb6a9cb7486049567c1c2f805e0d7335 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-16dbcb1e | sha256-ef7f717b2b4ef87a4448afcf5d328bf153cbfbb83527d7ad1115aa7903ca5b49 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-bde11863 | sha256-82929c1e645b698654bacc42cbc25583a794ba600e553bd9275caf8713b006dc |
