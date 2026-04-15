# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183`
- winner mode: `graph_prior_only`
- trace hash: `sha256-203ac39480367005fd42cf7825311a0cbe85dd80f56721c12d00a8ea3f270b1f`
- fixture hash: `sha256-f1b7e7068a4652fbad5d085cdb0c1a635468b0ae89cc507258e65b4da9413c08`
- score hash: `sha256-5299bf38f4dbfff42b13af3a766b9a013f53c8d03adf3bcabc851362fcfcbe15`
- bundle hash: `sha256-539fad38f54c88d21e4e07b63ab9d9621746469053cbf89090dce74cd8f9bfb1`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ec523fd6d1df556efd872a8228174bc9b8350cfadcd29169122cad9742544bc2 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c12ebfda47a5445dde106fa232b19b3844a3e415ddde42172b284ca89b703c66 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-07256c084ad07aac177b2520e8ca1a0df2a02cc0f51a6cfe9cafac98ef3fbde6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-454f8efb | sha256-70ccacd8cbf518bcc24243d48628266a7573cededddfe6633602626c09132508 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-454f8efb | sha256-538036b469a36689540b9439837a170d904e9285129df1eafe9f3bcf8e67408b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-454f8efb | sha256-70ccacd8cbf518bcc24243d48628266a7573cededddfe6633602626c09132508 |
