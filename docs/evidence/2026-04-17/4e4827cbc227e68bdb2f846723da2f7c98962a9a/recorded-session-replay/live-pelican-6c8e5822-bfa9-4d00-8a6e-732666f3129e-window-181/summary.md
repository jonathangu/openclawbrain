# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50cbfb4de0d67c0910ccd1f15acc9132454b767d6a9ef6092fa51c701d086751`
- fixture hash: `sha256-ed982aae33c06dfcffb629c09975a63d396b69570ab9ad349366a4a66aa757f2`
- score hash: `sha256-1e7a0178cc2d25b0a38005fbe3a8f2c084e9bd81fb7e6c6164e1ebf23e12d93e`
- bundle hash: `sha256-b2e172599c27ae330b7013c51c2a39115faf5575a16439f50fecb04a6074f264`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ce8acab11edc00f581b930ddd46ccaeed311548b8f75f0398d0e21fa5078567 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-aee252bd11b3defe61051cbb3109057b3f3b11e86ace12fdd3dc7cb5fce66918 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-c486ccfcbf794f2b3902b0e47255867efae50df29202e50e22ec903306741392 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-69c9c634ec097247b39f72c7486b8efeb6fa9ed5a5568c32cc9729476f574bfa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-b961b401 | sha256-ff20610ca25af5b1e04b9e6a987f0e20b0f70213c4370a17cdf696c7257ed4c4 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-b961b401 | sha256-a017983f8ff7b02ff51c769c546f3f1b27c1ad188a7bc5dbc3283fd1f07c92b0 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-0b1ad4ae | sha256-dcc35bd5e8edc6225499774181bf813821e723a723f6ab81175e2494e7423ec4 |
