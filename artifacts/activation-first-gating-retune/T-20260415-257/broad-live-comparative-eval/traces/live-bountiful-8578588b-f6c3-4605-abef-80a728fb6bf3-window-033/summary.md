# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e6da83215f56050459d7b523ac12aa0af75a3f0c2a58f526978f2254a29ebc94`
- fixture hash: `sha256-d4387ac5a22395546761e4051b3bacd61069a298c0e71126f4fabbe9ecc70ac1`
- score hash: `sha256-267bb6d2b2d701363916774e73dd54dc8605693624a19df6e8ce5bb6cc7c8fc5`
- bundle hash: `sha256-a7083426c37a6a2bcc1fba2ca8a38806b9c0869966a8f7a4fcf5f5c65c9d83c2`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a588b2ee957ec985d426ad6e49fbf57081b897d4927c49e13bf488a2002d7bc0 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3abd387dfdf950a150fc6579fe6c847a52a937733a7cd5c42f86a7ff5c1f279c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f93cf785d92a4ad637d1ead59cfdb00f27b0f704ce5736a8cce434dc0e80a24 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5f991aae6501cf332f34bfcd345e1499fbdfa4903b6cfb13a71e5b98f6b77e39 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-19a14a9c | sha256-2c0c0cdf71bcbe134533bb10bd656d9348f53b06de6ccd3e49ea1684b9b7caf4 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-19a14a9c | sha256-7df75c331e22327b78d2d97b4147462d58c14bb0cc4594cc88f56f66f6de5f3e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9e144a0b | sha256-f7c7ce16354e80eabf249a6ac6422fbd20e1c1b295671ecf68dd8a7305232f5e |
