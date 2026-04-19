# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a462611820be4f93d663b55c36826ca6c875b16681bbbac07441c4d98461cd86`
- fixture hash: `sha256-55dd2817c613b4842f9e8a859b558557a568175128bc3b05ebca7185c8b4c45a`
- score hash: `sha256-0131fe184aaa8d90986813481a15fed78f1b1f3c669f505acbf25d8f748b98a6`
- bundle hash: `sha256-4760fc810e060b4ac96fa77df429789026e7449f8fda4ec19f63edf4a4b7fcec`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74198a04279dede09056c15d496cb33d205dd223adf4c279bc3faec0cb8bc3dd |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-5cba9f99d916244ee2146b02dfd60953cc6eb66a638d4ae1809f69d31cf4f351 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-8d2a4e60aea5eb8a5578d6c691a98fad97cba861d19937084adbead60a57a047 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-ef6056fb7d895bf2320974d4f46867bb85ca1f99d1b5350251557e6935affa16 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-cacf2fa9 | sha256-d2343123fd02586d86288aa6a7e521e1c335e96cb49bcc3d456f217d51ca7ff0 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-cacf2fa9 | sha256-b033d509549d371d62bb05599c5ec5df7d9262ea3076bd1730ebc0181880b3fc |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-cacf2fa9 | sha256-1d7d3cd1fd5837a822306d9d3feac58354fedd5ffa7c3d47342d752f6225e689 |
