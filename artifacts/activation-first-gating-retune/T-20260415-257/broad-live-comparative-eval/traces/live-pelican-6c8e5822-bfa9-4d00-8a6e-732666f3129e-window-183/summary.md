# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183`
- winner mode: `graph_prior_only`
- trace hash: `sha256-203ac39480367005fd42cf7825311a0cbe85dd80f56721c12d00a8ea3f270b1f`
- fixture hash: `sha256-f1b7e7068a4652fbad5d085cdb0c1a635468b0ae89cc507258e65b4da9413c08`
- score hash: `sha256-393faa3f4afbfaeca19a43f318173e695dfffeedea77658342c69f47a54d33bf`
- bundle hash: `sha256-5e6686ddb8fb1c5ecb556dd55aec47baeccb2741235e5027f0a91bb5c62d4dd0`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-08044093faa209a549cf6cbe79d77a3fd872d3cdde2c86b5886da5044f650477 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ddbee49e54ce4aa3c0097f6633dafde0fa87aebcfc866c67a2cdeaeb184ae7b1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-747f8324145e14ca465322214f34413043a7b4070f9bbd8e4af780bee9af7ce2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-8d9e15fbb62a7ba994e1e1a964aa2a51f386cb3ddb90ab87a972406e4a9fbb0b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bc9fd485 | sha256-133010a51d88b7a7072699f376ec6daecb51b2696441ce385aad515333d7b071 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bc9fd485 | sha256-946b86cd8fb69ee203ae9b04575109a1405b77de5c90ab04ff8af97c7073ed4b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-2888946e | sha256-3c3f034ee4ab34271c901176d5863c647a5c9dd015c8d15db2d819c7e7bb8d4e |
