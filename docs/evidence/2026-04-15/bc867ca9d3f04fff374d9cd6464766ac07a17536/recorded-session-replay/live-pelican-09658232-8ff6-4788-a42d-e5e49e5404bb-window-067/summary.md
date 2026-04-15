# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b511b4ac1b0719d0780ec2c9e18278c04d48d286fc6710bb079c3e2eba6029e7`
- fixture hash: `sha256-801021da403c0e7ffabb1f8a7bb11de0378ca6fc16e45764ab572505b0e2f302`
- score hash: `sha256-c395f0cb503f09d3cd6af498b449ab0b452c8fff61e5bcda4b70cd2744746add`
- bundle hash: `sha256-2d20fcfe8ab8cf1ec2b1f3ce8edcea5b17bc0dea70aa2d5b0b1524526fee295d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88537cfda5a24580056c21d94f97ea80249672c6a52c1a8bd0de62e2aead80ad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c380808fb118d6664799d8c7864441826eac4e480705e96b1236439bbaf1c7f3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9c9515e4aa908458531b016995bec65e07323d17286c9730e6acd4dbe1947afc |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-080bb60db757c8589a015b1ab363d4b59bffcb972a7a2e7d8469cf30289f28a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6392ff9e | sha256-4b8ca3170466e7fea6fb385f24f28213f23a5a52ef7d816eceebb654a29415e5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6392ff9e | sha256-8e16c267dff1beb541aa88b44c338e23683b85f752ec22fc1191ac0cfc337e3e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-6392ff9e | sha256-4b8ca3170466e7fea6fb385f24f28213f23a5a52ef7d816eceebb654a29415e5 |
