# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-32e0b4ec2c1ecbf5a44b66dab5340f30730d05ccd8fc6dea8e459b03d93bb729`
- fixture hash: `sha256-cd231e74dab2c7ac691e39a4ea475c769c350fe4115dc674162e2af0c0f3148d`
- score hash: `sha256-b5f469560e354318728826df68b3274b377be4ae4b73c71e5c485db1942cf899`
- bundle hash: `sha256-f7b7be929995e9aa9adc1ffe1b39b78e2aec7b4d5ab783b7ba28b385a6d40561`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-eafd155fdfa2fbd8e1c5739855382bb4aee55ae760f037b37bc2cd66c8f2b4cb |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e241533431c0fc3aba12862fca7ee8c7152b504b9e49f205b83c2a38afaa629 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-f6d1b0bc6bb50df7a0f85e54f8b71c52aeb54b97a1880858149a234e3135b519 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-a14e8649ad291cc5a4b3b42329eeddb1020546521fb135dd3d903c17a1b3d573 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-272a2d73 | sha256-f525f19b0f1e617e1061536dff186d53348e86b962fc692d22dbed8fce8019c7 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-272a2d73 | sha256-fba3ea0750a4ef11a794dacd47a44d5cb2bab7b3f13870d1bf943cd8fe464a2b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-2e25866e | sha256-0822617ebc864390fc058ebb98488f6b95c7ba21fbd5777411936a3e030aaf09 |
