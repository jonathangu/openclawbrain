# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d958774a8fc5556f6b626cb2afd5141be38b390f01c3f1c481f5689e5c67765c`
- fixture hash: `sha256-bf711d8c588faf57d4df6088b8652fb030ca7a163bb118e31c3e2f2768cad0f2`
- score hash: `sha256-03183b4228e65566dcc47310adfd5207d15278b7cdc080533c178a0b08fd4d0d`
- bundle hash: `sha256-2e97f949852dadabee2f382ca6ec12799c2e467edf8bd62811954dce4a30e562`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f7ca30e9c8433554610f300b068b172fcd1c7c716d277545f4d5940081fb358 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c61dbdf33ae2c7c1fbb3e16c02454a8e0d980f5c2b0ecb5ffd7fb554f228f176 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b9bcd7589984ecd14a3c851ed07882b72e28eae5910aa910c45b0398473ba531 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8ff308e77fe961a2ad85e9874e98b16c695a320c082657ffd32fbc255c6bb3f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5d9e40c6 | sha256-38fd2e50a994e1886a4b634d7c75f445b3ab09626bb06ae36e1814a7ed1d243f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5d9e40c6 | sha256-e958695f1c3ec22190684fe52b95e4edc3e5f1f0a953af9033bfb657e1002f70 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-07fbd625 | sha256-6f8298c65d520f3d36d8ecad244ed5e0f2a552f7c94feacbc3d6a5f8d75c25df |
