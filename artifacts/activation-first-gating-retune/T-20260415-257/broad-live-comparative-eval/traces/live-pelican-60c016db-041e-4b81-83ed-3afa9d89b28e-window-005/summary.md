# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f4905564adc9cb953b8b5504309a4080c3ac583fe0f629cb62b1e05f91ea23a3`
- fixture hash: `sha256-0ad0b5e1e0f2271069ee0d118e38a8f083b22de4d11f9b10cb9ee63b3ed54883`
- score hash: `sha256-7e81889f0329ad1bd8f9e4a54c0ade5bb1b1cae6bc5f418a49fc0ce49aab273b`
- bundle hash: `sha256-da9409b7e2a8b754fc96e27f412ea928a43eafd05369e06c3e3d9102b73fad6d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-64badee520388e2e251dcf80ba87d74776085beb63219f4be30791f06cfae40c |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b4af856b8be39f467164d307ead72083ca0ed30ccca5a9e4fdf88ab30164e998 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a6f8d1e24bbebd607fa4f4ba1ef3a0015e4fafc69e299a1120a8ee6dd4640a96 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bf00c8e0ef00c2e67d489f7fe9dd66625bce37ac6207de2cac0645b30223a8de |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a3c8750c | sha256-3c005bc84917f47504ad084d76bb843bb34de5c260f2136adf96bb6bd7569500 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a3c8750c | sha256-a0299f4a1b2017ba0cf06ec7236033f8d1651de5d60688dfd051f03cf277b863 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-6b44ce99 | sha256-5e3b06d4d708da9653ed798eda8763609f66f1630b7ba7df7c7703506106917f |
