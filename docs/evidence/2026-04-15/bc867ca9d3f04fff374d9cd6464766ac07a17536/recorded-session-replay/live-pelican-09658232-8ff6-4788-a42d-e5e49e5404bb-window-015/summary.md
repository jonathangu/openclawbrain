# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015`
- winner mode: `learned_route`
- trace hash: `sha256-6e0ff46b57f7c50af20d23a4d8a3f648535a36cc4021c3842ecad10617366b5b`
- fixture hash: `sha256-d2c3dec4ca105c441633ffddbfc56cdd05e4790ddeac1ae7cb9c9e93a7fe734a`
- score hash: `sha256-3f2b6c3ab476bed157e8afc16fd6ccac15c37f1b774c06f04c5eed6e0df7f377`
- bundle hash: `sha256-cb7adf23e036b23e91c741f6766fc353c7bd2908255d4720cd7f01c552f68824`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-48620826d5383928480bcd6e76b64505c9f9a20a21654ee2da4ad581ffb660b0 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-998ef8650eaa519b4fa0e6e23c5173b0f302a11117cd385b56f2f4ce0f4dcd5a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5dd09b2a0327095701dd36269a48c6680d32517076e98c79fcf0f27eb7ad73ee |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-343c48c8f002fd4adee8fd4e659f2c7a02d4202662131e7e49450b64fcc2703e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-0adf9097 | sha256-d06f33eae48178fde78b2bc54f2ebf5f16098637c294bd8ded80975d3624559a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0adf9097 | sha256-58c7b1ad55c2087aead5af39a66529b4d418f2a24a6bfc3508f281c87a16bdb0 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-0adf9097 | sha256-d06f33eae48178fde78b2bc54f2ebf5f16098637c294bd8ded80975d3624559a |
