# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e1899273f160788957979a298d976827dfb7d2c8980b1c161a6e0c69b405f12f`
- fixture hash: `sha256-e3a4578dceff89673c40bbf12c9b294dd97be3ba2d82b9f266209970182a5648`
- score hash: `sha256-9e8485729fa236e8434a99b403827022d07fb42d14ee041a56c1a8dd51bbb306`
- bundle hash: `sha256-b5553c758c6695055535dc56cf25e2bfe229d4726b9caf265d08ae0dbb452ddd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ecf5a06a6508fbef20c40ee36944ffad441534c7ec83a389bf5c81a0f73bcb66 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f7ec61d527b06c846fb0178cccd54b2e9ec191c9b0258d5b714436ff5d66a5a5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a680845443ce0981ec307dfc9efcd0663fad87f45552c9d0ceb8702873ac0b75 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5f231b6db73a1184ba66ede326a26aebb5030b5344cf0e873e15ffa383ad24a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-170a66ec | sha256-64b8a8fa24da36bca56e5148dc3b1d064442686bb43723508e303c8ba619c8b9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-170a66ec | sha256-94f7232471cb06692f63fbd57403ebfe1e689391217f7e622e585705575d7e2e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-16ebfecf | sha256-e1e4d98d13da9827eb1786060a73583075b40501c02a4e81b7b29a793929b151 |
