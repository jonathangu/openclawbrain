# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1e91f891be11ad983e343a9bbb8eb7e094a3203fdeb0cba32d80844dcceadc5b`
- fixture hash: `sha256-c962d7bf59f91132e81f529b35b43a46128d3cc144f19a803783e383eb2588e0`
- score hash: `sha256-ad17654d9cd8b3c8fea407ce977aac4ec2a893816330c54ce7bef411fd79cbbc`
- bundle hash: `sha256-6ecd39afd00528fdb9e3987b5c60bde24b63cb698760f56fd75ad6e3c7ea454e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5abc89ba1c4aafac24d8b492241ea58c50f7925494e6166e3016c9a753e61584 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4c439eb1cabf0ff0e325fa00059dc670e8acb2f384971ccffecb71c253a8f38c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bd19fee83758c4e240b693f58f2dd1f83ac0bd232ceb7816ea03835645b89cbf |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3466055edea5a00111cd0dafc5cbb02f512a43aec16fcb54ebf52a6b1dc58bb8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33d7bab3 | sha256-8208c4d0562f99a10d985be35e5a88fadff35322fb18e0fb32c26ae3f4675b9e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33d7bab3 | sha256-290a0068181542638f82de6f43c861077a0997b204b729f068ce0cb1ecebf1ed |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3f4fea48 | sha256-1a48274b65c3f0f90e2f28bce40f0529851e026af2afc7423789dc0d50913a25 |
