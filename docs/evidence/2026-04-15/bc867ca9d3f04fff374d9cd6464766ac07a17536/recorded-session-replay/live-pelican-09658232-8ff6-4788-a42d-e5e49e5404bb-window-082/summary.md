# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bda6b3da4ef39b29be45310328eb0566a39d316769663e62675a6105dd7880f7`
- fixture hash: `sha256-e12b530a582d1487040cb7cdaf3e1255576e9298c334dbf79363d1f81080b1c8`
- score hash: `sha256-b92b6449350c6b299b7a699e2b166630da35ca9e5cc7f4b02f7109137ca008a8`
- bundle hash: `sha256-b5525a642ed3119fc6efb8b46968d2a4aeb8a4bc6108064862503e726c926dc0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bab067ff9232cc412579013f9b35dc498686eb53e7f83b8de58e12e80ba3c742 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d4fb1f5cd4e61197e3eef81d22161d97cddb3c4d24b7d3231cbb689cec6ba13c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-34a7bab970f7c1ca3d98f09e4f7138f50b0c77cb7a235f5cfe5d34e6ccfc9465 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-750e12c43190e573cb1be93e4dc79300c52da7e70eec1dc616bc606905877b78 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5a342490 | sha256-18fc42af00ad53e786d86abe72a40d49147e7edb470367159c330e279cd0e7bd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5a342490 | sha256-fbe31dca9f3b589d07c758d20ff9b8d19ea37cb911da438185cfb8b88e3b423a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5a342490 | sha256-18fc42af00ad53e786d86abe72a40d49147e7edb470367159c330e279cd0e7bd |
