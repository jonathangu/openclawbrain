# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015`
- winner mode: `vector_only`
- trace hash: `sha256-6e0ff46b57f7c50af20d23a4d8a3f648535a36cc4021c3842ecad10617366b5b`
- fixture hash: `sha256-d2c3dec4ca105c441633ffddbfc56cdd05e4790ddeac1ae7cb9c9e93a7fe734a`
- score hash: `sha256-f5b50af1cea34ec9bcaa7ca0445a2b4427020771a5bcb3130cb160bcad2dfbc7`
- bundle hash: `sha256-8d68a0ee603f4a935173555529735ff1fe3dbf1cd899c7a8ff2484151f82d98d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 100 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/4
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-48620826d5383928480bcd6e76b64505c9f9a20a21654ee2da4ad581ffb660b0 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-a22e2d06ca9efae208000f5084355b5ee724568ffe67c2fb74e4a64f825381a2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-18729509162e9a744eb5b5ec2cd1c760e78a3cabbf0b77116e377a84dd6059b7 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-53d92a5362612338872fbf61670bef5b431f0acffa6bb80be3fa9b819a18c10e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-8130a5ec | sha256-5e5f5591de682fb24efe368d87ec1be11bf6327e06b25a00e57e83f27fdc027b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8130a5ec | sha256-c5c0053d1167b43a676faa0d017e5177d96de0b6bed8aa27405bd7f52ca9829e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-df47344b | sha256-1df70b8ca4cc6471c0830105a7740b464f7088bf25f55343cefccd4485179d47 |
