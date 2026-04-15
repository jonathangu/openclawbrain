# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681abae2fa5f82c72e9292e394b227021bc61d412148159906a6b997f617cca5`
- fixture hash: `sha256-c46f6cfeb0331761c1d2bb543d4b028a9a876c69162435d955a285bd82156828`
- score hash: `sha256-dd619cae26a814229c52af2b2a151cb733469a255740b4ea2856492da9d2a9bc`
- bundle hash: `sha256-ef49f0327205d102f3cb7ba3b1a9478c04b38b9edc40b4bfd5ca94d3ed13f8e5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-461e5f784d6e942b4cdd1338a01f757f830996458c9f4abe17a0effaceafc63b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c369bcc7bcf5d964972efaf633c73438251600ad7bb4d0ef093d1c572cfed73a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-064f6369a4c697cda979f3e7d1800ebca4831dda9a6bf378a71532db5049570d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-882b71a646008f69376d3be490c46652e24fc18914e638a30489e197d6df053c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7d8d1aa8 | sha256-4412a15d14c8c9c24e8d1839e336759ff1f8cca360850f63243fd1eaf9cd0e9e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7d8d1aa8 | sha256-da37a9f59b490a2ce82d323ba6e52d0eeec4bc94020dc49f94b79c690ab8d666 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7d8d1aa8 | sha256-4412a15d14c8c9c24e8d1839e336759ff1f8cca360850f63243fd1eaf9cd0e9e |
