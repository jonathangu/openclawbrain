# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57d54d463c3d335756b9ef845ab48b21d6ba79bd455096740f5eec6ab5dcf52e`
- fixture hash: `sha256-c975a7548913ddc09f78bdcf8d6f035b2cb79bee5a8fff204c28b6e92be5b531`
- score hash: `sha256-b2dd7d9b119d57a119e057d58389e509a9171a8fa2b4ff31a6eb5629f0427a3f`
- bundle hash: `sha256-2649a6b1312c28a8400cf8d951f5dba63b18545c7e877d933b22ad0e09e51e07`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f565862470853eb1b48835f5dc58d5e78705c4b54f6971c4806d12966cc7447 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b160bdd186e569574950bf57ab370956dbf6827892c528dacfdcf6eca0c9a36f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bce011596b132d4744206983cb9a29666d0c6446de688e7953950bad39fbde57 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-cfee816f877eb06f53087c869fcf3e5d7f6f444cc8f239221127659e2ec84083 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d4e3da88 | sha256-f842e417b1c796e48d2a04722471b29ead378787212ae45d2460c82a75825bbd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d4e3da88 | sha256-7253d25b3a16e4716bf4f5765477890405ea11fe83a5abe57d453ca2199e1e47 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-60bf28b1 | sha256-a6cccc70b2497f430ca97068897832501076f0600989828beedc6bc7b98d8888 |
