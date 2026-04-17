# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b511b4ac1b0719d0780ec2c9e18278c04d48d286fc6710bb079c3e2eba6029e7`
- fixture hash: `sha256-801021da403c0e7ffabb1f8a7bb11de0378ca6fc16e45764ab572505b0e2f302`
- score hash: `sha256-74a8b83405a061b0566903d3802158cdd7fb10484282582c2b4ba2f5e9e54f30`
- bundle hash: `sha256-455e16c2233a019046ff24d372764d12b151f9ddd16512d380531458be320dda`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88537cfda5a24580056c21d94f97ea80249672c6a52c1a8bd0de62e2aead80ad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3ebf73526d0275b13cb6df22328018136ede7dc297a7b3aa3f2b818e6b5622a5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3a268f5e1f308d6abb01733e50610db8863b0e244ed7dfdfe6b7f031a9f340f5 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-551d8cef31695c811a7d242e72290a68dba883758d45b392c94fbf30d6392062 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c9a2fcf4 | sha256-09f14bb49cf43e0a5fe69c5bf41eb5d9a8ee7136ccc43d8a4194e8ef5202a349 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c9a2fcf4 | sha256-9615418a1c7820408a172d682477fd8b5fa654f323aa33e921a6f063b1f762e6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1e8c8e23 | sha256-e378d3bbde059b1096507514b095abb583b3473a6081cdac5222f70bcfe771f9 |
