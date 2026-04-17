# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f10c00dc4efb180b5273900a9e561d1c614344a77050359aaa2d54aa27cc20d2`
- fixture hash: `sha256-43065829df1e95ca79dff07d99e5773679b5561b6bbdd3945d317201ab2cca51`
- score hash: `sha256-f7d079c8c6f713910470e3d239fe42f215181e7b0d675135e0aeb11a2ee598b7`
- bundle hash: `sha256-23780c431752bb60f000233f3fc5b49cf6a81fcba5aa95a71118ffcd164ee057`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f1460fd13a644dccb389d5e4bb97bb20a28fa61d221da193a36a1bd2b7379c0d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-29b8588856f0361b643dc3b9c2c9b7b13556ebb02d9ee2340478cb4bbba2a63b |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f035a5be18d4be836876a6152bd79a2a56561168aa579c4e2068f5b4c3531c4d |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-e02b7071aef998f52da8d2d92be517e1073bab44beebfd87cee1a55f82c52073 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bc81ff4c | sha256-b5784b4d63ef3c42d738ce16a3c1b0c570d62ac17dfcfa6620c2cb58ae3586fa |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bc81ff4c | sha256-b5784b4d63ef3c42d738ce16a3c1b0c570d62ac17dfcfa6620c2cb58ae3586fa |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-78c19101 | sha256-8eebff7695e4b3d24fb60464477d17734f290305f8beba8127ae450f5ca30a95 |
