# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6225db554a4364f98319fe1020858f0210ac404609a5303b38b0c9b7d31d658`
- fixture hash: `sha256-ed619df46c192591981773c10de0d36b74c9e5dca7f78e3181e7b1aa8b066c66`
- score hash: `sha256-d408839de965d094417e60512631fdf78613805f210dca7a1e15048b32b54ed8`
- bundle hash: `sha256-58ccf26f41ac4cece494ad649e454824480c2873546344aa8e7ba2647b54f09e`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88c859d8e45ae0f9acc890628cb16be0104a115fdd6d7748c85da1fbfab29bae |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0a059bc5949b7c4c9d1587c7774d958f80284afcfab783a7ebf085affdef3c38 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-74f5d87e92b9d524ca12bbc223dd42652f30ba3acfb2de632a6042f065a4b62a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-df9d36ce9593ec4a37a877f32d540142b797a307171cbea53ae16421ac89069d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-de1fe319 | sha256-8669d4c1b7896b434fb3f4594c5225172ec9979afe6b76daa377530a69429280 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-de1fe319 | sha256-8669d4c1b7896b434fb3f4594c5225172ec9979afe6b76daa377530a69429280 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-de6ac500 | sha256-15b58c722162bb9b241ee741d4a2b5b7e8ba23773705ff11b38ae8af88248d20 |
