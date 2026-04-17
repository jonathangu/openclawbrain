# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0b1bca8dd8d311ca0f474a7d9deb1193514002f9ff0a549efbdfe8a579f7a8a7`
- fixture hash: `sha256-693be8683846991e932bfa4a0d12773f4fe199b9445b669c78493c22255f8959`
- score hash: `sha256-44c6a26cba7016490fd0c6097fa026f3673a90c51773c5e38085cdf47f6d92fa`
- bundle hash: `sha256-83d1304c19b5c2b80b0890afd5094e31664276886c3ff291387aac30f904d7d8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a936bbb2a6bddfe389caa1010c92a0418532436fd2f50651530e961a6495d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f4787587826782296dec73e76ba97db2ee0165a44f560b119f53cd66a9d7df63 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cf999a87169da99cc1ba61ec7a8c5b41f2ab84678aafe372fa59b5439710d267 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e2a3f401a20449950bfa52b7bdc97c67bc82dff6872dd83aa093bafa6808db63 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-108cd6ff | sha256-41c6bfec68b2ab7aa969bbd538b5a63b2200fae9233514fa6dc7420bee8de581 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-108cd6ff | sha256-5f417f5fa732e7aac740f049ed6af2b6dd10f144008ea8ba7b058181689e4647 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a20edfc8 | sha256-60ca4dd3941c5b035a56ccf4d322b75c2f264569b67f63e5ee22edf41a9566cc |
