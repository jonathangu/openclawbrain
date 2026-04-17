# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-154227e12deeada99188001de1f98c7859b44b0240a0a63280198f0600727836`
- fixture hash: `sha256-141e98c67b76e6b544c136b2dc9ec311316dae947241f48af13f9b3f509e9c48`
- score hash: `sha256-e78377bc5c2d5a2e4879e487452fb72cd9211efccdc3364709f87ef00567429a`
- bundle hash: `sha256-2c8c83a6f613b0742aef54a7d9027bad629194c4d065b676f4f445f49242c6ec`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fd20c45ec549a50541ad825ca2263c2905bab11bd8f991e3eba1789bd6eddad |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5e9005a3efc5691b01adef3b70e7e78f7d60ed93e8d430702b7af243c7c41fce |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6fbdd5625bce92fa15ddc84217ad399c50f365d02a8fa6fbf641d80dbff742d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-22e541886de4c4a4664cb1a8afe7acba40d28accf2c6c17b530657d8cecc508d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-69ad1fd9 | sha256-af8f8eb918f72f462cda4e5f858047b69a35214d5f8c1a85b1eba9791034e17b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-69ad1fd9 | sha256-0b2987785fe0c94ad95ec88010a070080acb7ab0e87a4f182987e7b257e2a9ff |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cfdc3100 | sha256-ebab40ece3303696aa4892b779f3fe95712b5b235a56e2270f1a6bfacf02f7ae |
