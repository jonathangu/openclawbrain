# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97f203f8cf6e54b5353937fa7c2a9de19dc80e0c9cd1f7a7efe923af5d952db1`
- fixture hash: `sha256-125dd958dc76d20d11fb8d8f175ae0fed91a68b28d91d5171d2943217403837a`
- score hash: `sha256-568646b7968c8e7f84c88a1bc44362a1d03fcc09f8c93fcbf52c53a715e7c0e1`
- bundle hash: `sha256-db173aef1ac426149d8a76187ee42cdda0679f94bb02668016eff46bfb4906e8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-511d0fcc64563746c6c18b192b94f28492b7e276c306dddc6df1e62d381fef89 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11fb53fa21f31f6b3dd1c370aad22001b19b730958a2c14037ac0d7770cdb993 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6c7ed89b73750d069e7fa627bf96f81b36f1230a176df7357b1ac64a78f685fe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cae0c89fd4bde149b2dbb70daaa5088480406441feaa5e88038b8862fb59e989 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9daf6a4f | sha256-d51c4ca34af61799fdab226b02514580a01e9ecf9167f0c351e8f285363eaa29 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9daf6a4f | sha256-6f789bc7b5246954dd72b7920fb4ba4c093c822d22983746b150905378a64bf8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-abe90930 | sha256-9e7e9c52d7b1408f48b9aa9a6ef5d625b412859fe593216559f79fce9bf2aa92 |
