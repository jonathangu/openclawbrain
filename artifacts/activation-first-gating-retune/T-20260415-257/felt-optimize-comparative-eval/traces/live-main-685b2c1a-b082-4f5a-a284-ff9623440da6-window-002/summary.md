# Recorded Session Replay Proof Bundle

- trace id: `live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6e4f37e5b65aeaf586e4d2a4e71ad3cdd999df7cfb455f741dd0b11a1a3ee8f9`
- fixture hash: `sha256-8f110e8a4894d421efbb2427ce24dd5ba84d98a2490639e91780761dd48a619e`
- score hash: `sha256-fc033554200f03a4802cc5cc3514072ed91ae0edc21e0bd54230794d3df6de92`
- bundle hash: `sha256-5d30e068129e88f724db3602968febce4781acd329b57f466f7a4ec3353e49c8`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb1995b9d3380942336af495a709abdc9277059dafdab742d11d02fd9c054a90 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-202153a47eff87a4cd46d4b1005d2f24c8cd9fa0269ca288c40b175dba4407e9 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-24d5fa6d97f5d81f979ec5cb019df8f214752a371420c601739645e962ecaac6 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-f8a8ab0e4a89561443509dc466b7fc50ac445a3dc85a8b1ab77838637297d11d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-eaa7c191 | sha256-1a6ae9288f1279e464aaa4dea131c1ab6bf55de36a7ca9916d8614404ff94860 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-eaa7c191 | sha256-50e8ee2ee8997fc359ca22466f25e882d596d238c95d90ff41aba1302629d80a |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-eaa7c191 | sha256-300f32e86388cfd81febd5dc3f9814844558ac05fb326a0070b4fc21c29c28c3 |
