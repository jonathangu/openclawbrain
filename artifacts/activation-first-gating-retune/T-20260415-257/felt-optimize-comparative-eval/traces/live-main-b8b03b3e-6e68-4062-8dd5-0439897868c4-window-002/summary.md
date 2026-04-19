# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b2fa5dad518df8f3866150e5eb5ae2df4d40d0d1730a7b39326babb425756a`
- fixture hash: `sha256-9c558cf390c2d5519271f6ba91a97c5aab0727de8cfbeaa1362c2e39d2a00c20`
- score hash: `sha256-1a189266ac781168c4954f1ecbfec1244692f7d8dc4765eef04f7497a36d9b18`
- bundle hash: `sha256-b86aa1f8e05ce3a59ecd78d4161e2ba907f60fc45013876d382e3bbee972a697`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8709c6589a780862225e4afa90cbbf44ed4ef4f7b39772bdc54c0a9f8a33087 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-b422f3578dc8eee43ce07aa875877e85fbdf246ce695f133dd0423e96b4a785a |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-c35a15cdd813f3fcbd185c4ec9ec871659328e37cb186a10393ff08f25be476a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-01acfff6c59d46533e7d42fad2f8e69ad684781de9f3dd69ab72e5896f2bcc66 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-fb5d4018 | sha256-390b83c74c61e8c63bfd53eb31f01f80411a1c4d066bad2f7cef5b0b0e9c5cce |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-fb5d4018 | sha256-029b7bc54a24ad6ce9b4922f3edc38285ce567a9dae13a4e335abb645a53ac99 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fb5d4018 | sha256-5aced15ffa5bcc8c537a10f6d0e3566a793ac23955d913d841c386f2654b5119 |
